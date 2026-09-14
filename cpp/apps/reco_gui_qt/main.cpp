#include "reco/core/path.hpp"
#include "reco/gui/app_model.hpp"
#include "reco/gui/gpu_preview_controller.hpp"
#include "reco/gui/runtime.hpp"
#include "reco/io/gstreamer.hpp"
#include "reco/io/video_probe_worker.hpp"
#include "rules_cc/cc/runfiles/runfiles.h"

#include <QApplication>
#include <QByteArray>
#include <QCloseEvent>
#include <QComboBox>
#include <QCoreApplication>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QFrame>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMainWindow>
#include <QMessageBox>
#include <QMetaObject>
#include <QProcess>
#include <QPushButton>
#include <QSlider>
#include <QStatusBar>
#include <QStyle>
#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace {

using rules_cc::cc::runfiles::Runfiles;

constexpr std::uint32_t kPreviewWidth = 1280;
constexpr std::uint32_t kPreviewHeight = 720;
constexpr int kSeekSliderMaximum = 1'000'000;
constexpr int kMaximumProcessDiagnosticBytes = 64 * 1024;
constexpr float kPi = 3.14159265358979323846F;

#if defined(_WIN32)
constexpr const char* kRecoExecutableName = "reco.exe";
#else
constexpr const char* kRecoExecutableName = "reco";
#endif

std::filesystem::path from_qstring(const QString& value) {
  return reco::core::path_from_utf8(value.toUtf8().toStdString());
}

QString to_qstring(const std::filesystem::path& value) {
  return QString::fromUtf8(reco::core::path_to_utf8(value).c_str());
}

std::optional<std::filesystem::path>
existing_absolute_executable(const std::filesystem::path& candidate) {
  if (candidate.empty()) {
    return std::nullopt;
  }
  const QFileInfo info(to_qstring(candidate));
  if (!info.exists() || !info.isFile() || !info.isExecutable()) {
    return std::nullopt;
  }
  return from_qstring(info.absoluteFilePath()).lexically_normal();
}

std::optional<std::filesystem::path>
resolve_reco_executable(const std::filesystem::path& application_path) {
  if (const char* configured = std::getenv("RECO_CLI");
      configured != nullptr && configured[0] != '\0') {
    return existing_absolute_executable(configured);
  }

  if (auto sibling =
          existing_absolute_executable(application_path.parent_path() / kRecoExecutableName);
      sibling.has_value()) {
    return sibling;
  }

  std::string runfiles_error;
  std::unique_ptr<Runfiles> runfiles(Runfiles::Create(reco::core::path_to_utf8(application_path),
                                                      BAZEL_CURRENT_REPOSITORY, &runfiles_error));
  if (runfiles != nullptr) {
    const auto logical_path =
        std::string("reco_video_stitcher/cpp/apps/reco_cli/") + kRecoExecutableName;
    if (auto deployed = existing_absolute_executable(
            reco::core::path_from_utf8(runfiles->Rlocation(logical_path)));
        deployed.has_value()) {
      return deployed;
    }
  }

  return existing_absolute_executable(application_path.parent_path() / ".." / "reco_cli" /
                                      kRecoExecutableName);
}

struct PathPicker {
  QLineEdit* edit = nullptr;
  QPushButton* button = nullptr;
};

PathPicker add_path_picker(QWidget* parent, QFormLayout* form, const QString& label,
                           const QString& caption, const QString& filter,
                           const std::function<void(std::filesystem::path)>& selected) {
  auto* row = new QWidget(parent);
  auto* layout = new QHBoxLayout(row);
  layout->setContentsMargins(0, 0, 0, 0);
  layout->setSpacing(6);
  auto* edit = new QLineEdit(row);
  auto* button = new QPushButton(row);
  button->setIcon(parent->style()->standardIcon(QStyle::SP_DialogOpenButton));
  button->setToolTip("Browse");
  button->setFixedWidth(34);
  layout->addWidget(edit, 1);
  layout->addWidget(button);
  QObject::connect(button, &QPushButton::clicked, parent, [edit, caption, filter, selected] {
    const QString path = QFileDialog::getOpenFileName(edit, caption, {}, filter);
    if (path.isEmpty()) {
      return;
    }
    edit->setText(path);
    selected(from_qstring(path));
  });
  QObject::connect(edit, &QLineEdit::editingFinished, parent,
                   [edit, selected] { selected(from_qstring(edit->text())); });
  form->addRow(label, row);
  return {.edit = edit, .button = button};
}

void append_bounded(QByteArray* destination, QByteArray value) {
  destination->append(std::move(value));
  if (destination->size() > kMaximumProcessDiagnosticBytes) {
    destination->remove(0, destination->size() - kMaximumProcessDiagnosticBytes);
  }
}

std::uint64_t slider_to_frame(int value, std::uint64_t total_frames) {
  if (total_frames <= 1U || value <= 0) {
    return 0U;
  }
  if (value >= kSeekSliderMaximum) {
    return total_frames - 1U;
  }
  const auto last_frame = total_frames - 1U;
  const auto scale = static_cast<std::uint64_t>(kSeekSliderMaximum);
  const auto slider = static_cast<std::uint64_t>(value);
  return (last_frame / scale) * slider + ((last_frame % scale) * slider) / scale;
}

int frame_to_slider(std::uint64_t frame, std::uint64_t total_frames) {
  if (total_frames <= 1U || frame == 0U) {
    return 0;
  }
  const auto last_frame = total_frames - 1U;
  if (frame >= last_frame) {
    return kSeekSliderMaximum;
  }
  const long double fraction =
      static_cast<long double>(frame) / static_cast<long double>(last_frame);
  return static_cast<int>(std::llround(fraction * kSeekSliderMaximum));
}

class MainWindow final : public QMainWindow {
public:
  explicit MainWindow(std::filesystem::path application_path)
      : application_path_(std::move(application_path)), model_(reco::gui::GuiSettings::load()) {
    setWindowTitle("Reco Video Stitcher");
    resize(1180, 760);

    auto* root = new QWidget(this);
    auto* outer = new QHBoxLayout(root);
    outer->setContentsMargins(10, 10, 10, 10);
    outer->setSpacing(10);

    auto* controls = new QWidget(root);
    controls->setMinimumWidth(360);
    controls->setMaximumWidth(430);
    auto* controls_layout = new QVBoxLayout(controls);
    controls_layout->setContentsMargins(0, 0, 0, 0);
    controls_layout->setSpacing(10);

    sources_box_ = new QGroupBox("Sources", controls);
    auto* files_form = new QFormLayout(sources_box_);
    left_ = add_path_picker(sources_box_, files_form, "Left", "Choose left video",
                            "Video files (*)", [this](std::filesystem::path path) {
                              model_.set_left(std::move(path));
                              reload_preview();
                            });
    right_ = add_path_picker(sources_box_, files_form, "Right", "Choose right video",
                             "Video files (*)", [this](std::filesystem::path path) {
                               model_.set_right(std::move(path));
                               reload_preview();
                             });
    calibration_ =
        add_path_picker(sources_box_, files_form, "Calibration", "Choose calibration JSON",
                        "JSON files (*.json)", [this](std::filesystem::path path) {
                          model_.set_calibration(std::move(path));
                          reload_preview();
                        });
    controls_layout->addWidget(sources_box_);

    preview_box_ = new QGroupBox("View", controls);
    auto* preview_form = new QFormLayout(preview_box_);
    yaw_ = make_angle_control(preview_box_, -180.0, 180.0, 1.0, 0.0);
    pitch_ = make_angle_control(preview_box_, -89.0, 89.0, 1.0, 0.0);
    fov_ = make_angle_control(preview_box_, 20.0, 150.0, 1.0, 75.0);
    preview_form->addRow("Yaw", yaw_);
    preview_form->addRow("Pitch", pitch_);
    preview_form->addRow("FOV", fov_);

    auto* transport = new QWidget(preview_box_);
    auto* transport_layout = new QHBoxLayout(transport);
    transport_layout->setContentsMargins(0, 0, 0, 0);
    transport_layout->setSpacing(6);
    play_ = new QPushButton(transport);
    play_->setIcon(style()->standardIcon(QStyle::SP_MediaPlay));
    play_->setToolTip("Play");
    play_->setFixedSize(34, 30);
    seek_ = new QSlider(Qt::Horizontal, transport);
    seek_->setRange(0, kSeekSliderMaximum);
    seek_->setTracking(false);
    transport_layout->addWidget(play_);
    transport_layout->addWidget(seek_, 1);
    preview_form->addRow("Timeline", transport);

    blend_ = new QDoubleSpinBox(preview_box_);
    blend_->setRange(0.0, 0.3);
    blend_->setSingleStep(0.01);
    blend_->setDecimals(2);
    blend_->setValue(model_.preview().blend_width);
    preview_form->addRow("Blend", blend_);
    controls_layout->addWidget(preview_box_);

    export_box_ = new QGroupBox("Export", controls);
    auto* export_form = new QFormLayout(export_box_);
    codec_ = new QComboBox(export_box_);
    codec_->addItems({"h264", "hevc", "av1"});
    codec_->setCurrentText(QString::fromStdString(model_.settings().default_codec));
    quality_ = new QComboBox(export_box_);
    quality_->addItems({"fast", "balanced", "high"});
    quality_->setCurrentText(QString::fromStdString(model_.settings().default_quality));
    export_form->addRow("Codec", codec_);
    export_form->addRow("Quality", quality_);
    export_ = new QPushButton("Export", export_box_);
    export_->setIcon(style()->standardIcon(QStyle::SP_DialogSaveButton));
    export_form->addRow({}, export_);
    controls_layout->addWidget(export_box_);
    controls_layout->addStretch(1);

    preview_surface_ = new QFrame(root);
    preview_surface_->setAttribute(Qt::WA_NativeWindow);
    preview_surface_->setFrameShape(QFrame::StyledPanel);
    preview_surface_->setMinimumSize(640, 360);
    preview_surface_->setStyleSheet("QFrame { background: #101214; }");
    auto* preview_layout = new QVBoxLayout(preview_surface_);
    preview_layout->setContentsMargins(0, 0, 0, 0);
    status_ = new QLabel("Select left and right videos and a calibration", preview_surface_);
    status_->setAlignment(Qt::AlignCenter);
    status_->setWordWrap(true);
    status_->setMargin(24);
    status_->setStyleSheet("QLabel { color: #f2f4f5; background: #101214; }");
    preview_layout->addWidget(status_, 1);

    outer->addWidget(controls);
    outer->addWidget(preview_surface_, 1);
    setCentralWidget(root);
    statusBar();

    export_process_ = new QProcess(this);
    export_process_->setProcessChannelMode(QProcess::SeparateChannels);
    connect_controls();
    connect_export_process();
    update_controls();
    QTimer::singleShot(0, this, [this] { reload_preview(); });
  }

  ~MainWindow() override { shutdown(); }

protected:
  void closeEvent(QCloseEvent* event) override {
    shutdown();
    event->accept();
  }

private:
  static QDoubleSpinBox* make_angle_control(QWidget* parent, double minimum, double maximum,
                                            double step, double value) {
    auto* control = new QDoubleSpinBox(parent);
    control->setRange(minimum, maximum);
    control->setSingleStep(step);
    control->setDecimals(1);
    control->setSuffix(" deg");
    control->setValue(value);
    return control;
  }

  void connect_controls() {
    const auto viewport_changed = [this](double) { apply_viewport(); };
    QObject::connect(yaw_, &QDoubleSpinBox::valueChanged, this, viewport_changed);
    QObject::connect(pitch_, &QDoubleSpinBox::valueChanged, this, viewport_changed);
    QObject::connect(fov_, &QDoubleSpinBox::valueChanged, this, viewport_changed);
    QObject::connect(blend_, &QDoubleSpinBox::editingFinished, this, [this] {
      model_.set_default_blend_width(static_cast<float>(blend_->value()));
      model_.settings().save();
      reload_preview();
    });
    QObject::connect(play_, &QPushButton::clicked, this, [this] { toggle_playback(); });
    QObject::connect(seek_, &QSlider::sliderPressed, this, [this] { seek_dragging_ = true; });
    QObject::connect(seek_, &QSlider::sliderReleased, this, [this] {
      seek_dragging_ = false;
      seek_to_slider_position();
    });
    QObject::connect(codec_, &QComboBox::currentTextChanged, this, [this](const QString& value) {
      model_.set_default_codec(value.toStdString());
      model_.settings().save();
    });
    QObject::connect(quality_, &QComboBox::currentTextChanged, this, [this](const QString& value) {
      model_.set_default_quality(value.toStdString());
      model_.settings().save();
    });
    QObject::connect(export_, &QPushButton::clicked, this, [this] {
      if (export_running_) {
        cancel_export();
      } else {
        start_export();
      }
    });
  }

  void connect_export_process() {
    QObject::connect(export_process_, &QProcess::readyReadStandardOutput, this, [this] {
      append_bounded(&export_stdout_, export_process_->readAllStandardOutput());
    });
    QObject::connect(export_process_, &QProcess::readyReadStandardError, this, [this] {
      append_bounded(&export_stderr_, export_process_->readAllStandardError());
    });
    QObject::connect(
        export_process_,
        static_cast<void (QProcess::*)(int, QProcess::ExitStatus)>(&QProcess::finished), this,
        [this](int exit_code, QProcess::ExitStatus status) { finish_export(exit_code, status); });
    QObject::connect(export_process_, &QProcess::errorOccurred, this,
                     [this](QProcess::ProcessError error) {
                       if (error == QProcess::FailedToStart && export_running_) {
                         append_bounded(&export_stderr_, export_process_->errorString().toUtf8());
                         finish_export(-1, QProcess::CrashExit);
                       }
                     });
  }

  void reload_preview() {
    if (export_running_ || shutting_down_) {
      return;
    }
    stop_preview();
    pending_seek_frame_.reset();
    runtime_probe_ = {};
    if (reco::gui::preview_runtime_probe_required(model_.files())) {
      runtime_probe_ = reco::gui::probe_gui_runtime();
    }
    preview_readiness_ = reco::gui::evaluate_preview_runtime(model_.files(), runtime_probe_);
    model_.set_gpu_ready(false);
    if (!preview_readiness_.ready) {
      show_overlay(QString::fromStdString(preview_readiness_.detail));
      statusBar()->showMessage("Preview unavailable");
      update_controls();
      return;
    }

    const auto probe_worker = reco::io::resolve_deployed_video_probe_worker(application_path_);
    if (!probe_worker.has_value()) {
      show_overlay("GPU video probe worker was not found");
      statusBar()->showMessage("Preview unavailable");
      update_controls();
      return;
    }

    show_overlay("Starting GPU preview");
    try {
      reco::gui::GpuPreviewControllerConfig config{
          .left_path = *model_.files().left,
          .right_path = *model_.files().right,
          .calibration_path = *model_.files().calibration,
          .probe_worker_path = *probe_worker,
          .native_window_handle = static_cast<std::uintptr_t>(preview_surface_->winId()),
          .output_width = kPreviewWidth,
          .output_height = kPreviewHeight,
          .sink = reco::io::detect_capture_platform() == reco::io::CapturePlatform::Jetson
                      ? reco::io::GpuPreviewSink::Nvidia3d
                      : reco::io::GpuPreviewSink::NvidiaEgl,
          .device_ordinal = 0,
          .surface_pool_capacity = 3,
          .decode_queue_capacity = 4,
          .blend_width_override = model_.preview().blend_width,
          .start_playing = model_.preview().playing,
      };
      const auto generation = controller_generation_;
      controller_ = std::make_unique<reco::gui::GpuPreviewController>(
          std::move(config), [this, generation](reco::gui::GpuPreviewControllerSnapshot snapshot) {
            QMetaObject::invokeMethod(
                this,
                [this, generation, snapshot = std::move(snapshot)] {
                  if (generation == controller_generation_) {
                    apply_snapshot(snapshot);
                  }
                },
                Qt::QueuedConnection);
          });
      apply_viewport();
      controller_->start();
      statusBar()->showMessage("Preview starting");
    } catch (const std::exception& error) {
      controller_.reset();
      show_overlay(QString::fromUtf8(error.what()));
      statusBar()->showMessage("Preview failed");
    }
    update_controls();
  }

  void stop_preview() noexcept {
    ++controller_generation_;
    if (controller_) {
      controller_->set_notification_callback({});
      controller_->stop();
      controller_.reset();
    }
    latest_snapshot_.reset();
    model_.set_gpu_ready(false);
  }

  void apply_viewport() {
    auto controls = model_.preview();
    controls.yaw = static_cast<float>(yaw_->value());
    controls.pitch = static_cast<float>(pitch_->value());
    controls.fov_degrees = static_cast<float>(fov_->value());
    model_.set_preview_controls(controls);
    if (controller_) {
      controller_->set_viewport({.yaw_radians = controls.yaw * kPi / 180.0F,
                                 .pitch_radians = controls.pitch * kPi / 180.0F,
                                 .fov_degrees = controls.fov_degrees});
    }
  }

  void apply_snapshot(const reco::gui::GpuPreviewControllerSnapshot& snapshot) {
    latest_snapshot_ = snapshot;
    model_.set_gpu_ready(snapshot.ready);
    auto controls = model_.preview();
    controls.playing = snapshot.playing;
    model_.set_preview_controls(controls);

    if (snapshot.current_frame.has_value() && pending_seek_frame_.has_value() &&
        *snapshot.current_frame == *pending_seek_frame_) {
      pending_seek_frame_.reset();
    }
    if (!seek_dragging_ && snapshot.total_frames != 0U) {
      const auto displayed_frame = pending_seek_frame_.value_or(snapshot.current_frame.value_or(0));
      seek_->setValue(frame_to_slider(displayed_frame, snapshot.total_frames));
    }

    const auto state_name = reco::gui::gpu_preview_controller_state_name(snapshot.state);
    statusBar()->showMessage(
        "Preview: " + QString::fromUtf8(state_name.data(), static_cast<int>(state_name.size())));
    if (snapshot.state == reco::gui::GpuPreviewControllerState::Error) {
      show_overlay(QString::fromStdString(snapshot.error));
    } else if (snapshot.presented_frames != 0U) {
      status_->hide();
    } else if (snapshot.state == reco::gui::GpuPreviewControllerState::Starting) {
      show_overlay("Starting GPU preview");
    } else if (snapshot.state == reco::gui::GpuPreviewControllerState::EndOfStream) {
      show_overlay("End of video");
    } else if (snapshot.state == reco::gui::GpuPreviewControllerState::Paused) {
      show_overlay("Paused");
    }
    update_controls();
  }

  void toggle_playback() {
    if (!controller_ || !latest_snapshot_.has_value()) {
      return;
    }
    try {
      if (latest_snapshot_->playing) {
        controller_->pause();
      } else {
        if (latest_snapshot_->end_of_stream) {
          pending_seek_frame_ = 0U;
          controller_->seek(0U);
        }
        controller_->play();
      }
    } catch (const std::exception& error) {
      show_overlay(QString::fromUtf8(error.what()));
      statusBar()->showMessage("Preview control failed");
    }
  }

  void seek_to_slider_position() {
    if (!controller_ || !latest_snapshot_.has_value() || !latest_snapshot_->ready ||
        latest_snapshot_->total_frames == 0U) {
      return;
    }
    try {
      const auto frame = slider_to_frame(seek_->value(), latest_snapshot_->total_frames);
      pending_seek_frame_ = frame;
      controller_->seek(frame);
    } catch (const std::exception& error) {
      pending_seek_frame_.reset();
      show_overlay(QString::fromUtf8(error.what()));
      statusBar()->showMessage("Seek failed");
    }
  }

  void start_export() {
    const QString output = QFileDialog::getSaveFileName(this, "Choose export path", {},
                                                        "Video files (*.mp4 *.mkv);;All files (*)");
    if (output.isEmpty()) {
      return;
    }
    const auto request = model_.export_request(from_qstring(output));
    if (!request.has_value()) {
      QMessageBox::warning(this, "Export unavailable", "GPU preview is not ready");
      return;
    }
    const auto reco_executable = resolve_reco_executable(application_path_);
    if (!reco_executable.has_value()) {
      QMessageBox::critical(this, "Export failed", "The deployed reco executable was not found");
      return;
    }

    stop_preview();
    export_running_ = true;
    export_cancelled_ = false;
    export_stdout_.clear();
    export_stderr_.clear();
    active_export_output_ = request->output;
    show_overlay("Exporting video");
    statusBar()->showMessage("Exporting video");
    update_controls();

    QStringList arguments{
        "stitch",
        to_qstring(request->left),
        to_qstring(request->right),
        "-c",
        to_qstring(request->calibration),
        "-o",
        to_qstring(request->output),
        "--codec",
        QString::fromStdString(request->codec),
        "--quality",
        QString::fromStdString(request->quality),
        "--blend",
        QString::number(static_cast<double>(request->blend_width), 'g', 9),
    };
    if (request->ai_model.has_value()) {
      arguments << "--model" << to_qstring(*request->ai_model);
    }
    export_process_->setProgram(to_qstring(*reco_executable));
    export_process_->setArguments(arguments);
    export_process_->start();
  }

  void cancel_export() {
    if (!export_running_) {
      return;
    }
    export_cancelled_ = true;
    export_->setEnabled(false);
    export_->setText("Stopping...");
    export_process_->terminate();
    QTimer::singleShot(2000, this, [this] {
      if (export_running_ && export_process_->state() != QProcess::NotRunning) {
        export_process_->kill();
      }
    });
  }

  void finish_export(int exit_code, QProcess::ExitStatus exit_status) {
    if (!export_running_) {
      return;
    }
    append_bounded(&export_stdout_, export_process_->readAllStandardOutput());
    append_bounded(&export_stderr_, export_process_->readAllStandardError());
    const bool succeeded = exit_status == QProcess::NormalExit && exit_code == 0;
    const bool cancelled = export_cancelled_;
    export_running_ = false;
    export_cancelled_ = false;
    update_controls();

    if (cancelled) {
      statusBar()->showMessage("Export canceled", 5000);
    } else if (succeeded) {
      statusBar()->showMessage("Export complete: " + to_qstring(active_export_output_), 8000);
    } else {
      QByteArray detail = export_stderr_.isEmpty() ? export_stdout_ : export_stderr_;
      if (detail.size() > 8192) {
        detail = detail.right(8192);
      }
      const auto message =
          detail.isEmpty() ? QString("Export process failed") : QString::fromUtf8(detail).trimmed();
      QMessageBox::critical(this, "Export failed", message);
      statusBar()->showMessage("Export failed", 8000);
    }
    if (!shutting_down_) {
      reload_preview();
    }
  }

  void show_overlay(const QString& text) {
    status_->setText(text);
    status_->show();
  }

  void update_controls() {
    sources_box_->setEnabled(!export_running_);
    preview_box_->setEnabled(!export_running_);
    codec_->setEnabled(!export_running_);
    quality_->setEnabled(!export_running_);

    const bool preview_ready = latest_snapshot_.has_value() && latest_snapshot_->ready;
    play_->setEnabled(!export_running_ && preview_ready);
    seek_->setEnabled(!export_running_ && preview_ready && latest_snapshot_->total_frames != 0U);
    if (latest_snapshot_.has_value() && latest_snapshot_->playing) {
      play_->setIcon(style()->standardIcon(QStyle::SP_MediaPause));
      play_->setToolTip("Pause");
    } else {
      play_->setIcon(style()->standardIcon(QStyle::SP_MediaPlay));
      play_->setToolTip("Play");
    }

    export_->setEnabled(export_running_ || model_.can_export());
    export_->setText(export_running_ ? "Cancel Export" : "Export");
    export_->setIcon(style()->standardIcon(export_running_ ? QStyle::SP_BrowserStop
                                                           : QStyle::SP_DialogSaveButton));
  }

  void shutdown() noexcept {
    if (shutting_down_) {
      return;
    }
    shutting_down_ = true;
    stop_preview();
    if (export_process_->state() != QProcess::NotRunning) {
      export_process_->kill();
      (void)export_process_->waitForFinished(2000);
    }
  }

  std::filesystem::path application_path_;
  reco::gui::GuiAppModel model_;
  reco::gui::GuiRuntimeProbe runtime_probe_;
  reco::gui::PreviewRuntimeReadiness preview_readiness_;
  std::unique_ptr<reco::gui::GpuPreviewController> controller_;
  std::optional<reco::gui::GpuPreviewControllerSnapshot> latest_snapshot_;
  std::optional<std::uint64_t> pending_seek_frame_;
  std::uint64_t controller_generation_ = 0;

  PathPicker left_;
  PathPicker right_;
  PathPicker calibration_;
  QGroupBox* sources_box_ = nullptr;
  QGroupBox* preview_box_ = nullptr;
  QGroupBox* export_box_ = nullptr;
  QDoubleSpinBox* yaw_ = nullptr;
  QDoubleSpinBox* pitch_ = nullptr;
  QDoubleSpinBox* fov_ = nullptr;
  QDoubleSpinBox* blend_ = nullptr;
  QPushButton* play_ = nullptr;
  QSlider* seek_ = nullptr;
  QComboBox* codec_ = nullptr;
  QComboBox* quality_ = nullptr;
  QPushButton* export_ = nullptr;
  QFrame* preview_surface_ = nullptr;
  QLabel* status_ = nullptr;
  QProcess* export_process_ = nullptr;

  QByteArray export_stdout_;
  QByteArray export_stderr_;
  std::filesystem::path active_export_output_;
  bool seek_dragging_ = false;
  bool export_running_ = false;
  bool export_cancelled_ = false;
  bool shutting_down_ = false;
};

} // namespace

int main(int argc, char** argv) {
  QApplication app(argc, argv);
  MainWindow window(from_qstring(QCoreApplication::applicationFilePath()));
  window.show();
  return QApplication::exec();
}
