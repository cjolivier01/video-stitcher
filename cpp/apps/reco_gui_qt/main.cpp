#include "reco/gui/app_model.hpp"
#include "reco/gui/runtime.hpp"

#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFormLayout>
#include <QFrame>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMainWindow>
#include <QMessageBox>
#include <QPushButton>
#include <QSlider>
#include <QStatusBar>
#include <QVBoxLayout>
#include <QWidget>

#include <filesystem>
#include <functional>
#include <string>
#include <utility>

namespace {

std::filesystem::path from_qstring(const QString& value) {
  return std::filesystem::path(value.toStdString());
}

QLineEdit* add_path_picker(QWidget* parent, QFormLayout* form, const QString& label,
                           const QString& caption,
                           const std::function<void(std::filesystem::path)>& selected) {
  auto* row = new QWidget(parent);
  auto* layout = new QHBoxLayout(row);
  layout->setContentsMargins(0, 0, 0, 0);
  layout->setSpacing(6);
  auto* edit = new QLineEdit(row);
  auto* button = new QPushButton("Browse", row);
  layout->addWidget(edit, 1);
  layout->addWidget(button);
  QObject::connect(button, &QPushButton::clicked, parent, [edit, caption, selected] {
    const QString path = QFileDialog::getOpenFileName(edit, caption);
    if (path.isEmpty()) {
      return;
    }
    edit->setText(path);
    selected(from_qstring(path));
  });
  QObject::connect(edit, &QLineEdit::editingFinished, parent,
                   [edit, selected] { selected(from_qstring(edit->text())); });
  form->addRow(label, row);
  return edit;
}

class MainWindow final : public QMainWindow {
public:
  MainWindow() {
    setWindowTitle("Reco Video Stitcher");
    resize(1120, 720);

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

    auto* files_box = new QGroupBox("Sources", controls);
    auto* files_form = new QFormLayout(files_box);
    left_ = add_path_picker(files_box, files_form, "Left", "Choose left video",
                            [this](std::filesystem::path path) {
                              model_.set_left(std::move(path));
                              try_initialize_gpu();
                              refresh();
                            });
    right_ = add_path_picker(files_box, files_form, "Right", "Choose right video",
                             [this](std::filesystem::path path) {
                               model_.set_right(std::move(path));
                               try_initialize_gpu();
                               refresh();
                             });
    calibration_ = add_path_picker(files_box, files_form, "Calibration", "Choose calibration JSON",
                                   [this](std::filesystem::path path) {
                                     model_.set_calibration(std::move(path));
                                     try_initialize_gpu();
                                     refresh();
                                   });
    controls_layout->addWidget(files_box);

    auto* preview_box = new QGroupBox("Preview", controls);
    auto* preview_form = new QFormLayout(preview_box);
    fov_ = new QSlider(Qt::Horizontal, preview_box);
    fov_->setRange(20, 150);
    fov_->setValue(75);
    preview_form->addRow("FOV", fov_);
    blend_ = new QDoubleSpinBox(preview_box);
    blend_->setRange(0.0, 0.3);
    blend_->setSingleStep(0.01);
    blend_->setDecimals(2);
    blend_->setValue(model_.preview().blend_width);
    preview_form->addRow("Blend", blend_);
    play_ = new QCheckBox("Playing", preview_box);
    preview_form->addRow("", play_);
    controls_layout->addWidget(preview_box);

    auto* export_box = new QGroupBox("Export", controls);
    auto* export_form = new QFormLayout(export_box);
    codec_ = new QComboBox(export_box);
    codec_->addItems({"h264", "hevc", "av1"});
    codec_->setCurrentText(QString::fromStdString(model_.settings().default_codec));
    quality_ = new QComboBox(export_box);
    quality_->addItems({"fast", "balanced", "high"});
    quality_->setCurrentText(QString::fromStdString(model_.settings().default_quality));
    export_form->addRow("Codec", codec_);
    export_form->addRow("Quality", quality_);
    export_ = new QPushButton("Export", export_box);
    export_form->addRow("", export_);
    controls_layout->addWidget(export_box);
    controls_layout->addStretch(1);

    preview_surface_ = new QFrame(root);
    preview_surface_->setFrameShape(QFrame::StyledPanel);
    preview_surface_->setMinimumSize(640, 360);
    auto* preview_layout = new QVBoxLayout(preview_surface_);
    preview_layout->setContentsMargins(0, 0, 0, 0);
    status_ = new QLabel(preview_surface_);
    status_->setAlignment(Qt::AlignCenter);
    status_->setWordWrap(true);
    preview_layout->addWidget(status_, 1);

    outer->addWidget(controls);
    outer->addWidget(preview_surface_, 1);
    setCentralWidget(root);
    statusBar();

    QObject::connect(fov_, &QSlider::valueChanged, this, [this](int value) {
      auto controls = model_.preview();
      controls.fov_degrees = static_cast<float>(value);
      model_.set_preview_controls(controls);
      refresh();
    });
    QObject::connect(blend_, &QDoubleSpinBox::valueChanged, this, [this](double value) {
      model_.set_default_blend_width(static_cast<float>(value));
      model_.settings().save();
      refresh();
    });
    QObject::connect(play_, &QCheckBox::toggled, this, [this](bool checked) {
      auto controls = model_.preview();
      controls.playing = checked;
      model_.set_preview_controls(controls);
      refresh();
    });
    QObject::connect(codec_, &QComboBox::currentTextChanged, this, [this](const QString& value) {
      model_.set_default_codec(value.toStdString());
      model_.settings().save();
      refresh();
    });
    QObject::connect(quality_, &QComboBox::currentTextChanged, this, [this](const QString& value) {
      model_.set_default_quality(value.toStdString());
      model_.settings().save();
      refresh();
    });
    QObject::connect(export_, &QPushButton::clicked, this, [this] {
      const QString path = QFileDialog::getSaveFileName(this, "Choose export path");
      if (path.isEmpty()) {
        return;
      }
      const auto request = model_.export_request(from_qstring(path));
      if (!request.has_value()) {
        QMessageBox::warning(this, "Export unavailable",
                             "GPU preview must be ready before export.");
        return;
      }
      QMessageBox::information(this, "Export unavailable",
                               "GPU export worker execution is not ported in this stage.");
    });

    try_initialize_gpu();
    refresh();
  }

private:
  void try_initialize_gpu() {
    runtime_probe_ = reco::gui::GuiRuntimeProbe{};
    if (reco::gui::preview_runtime_probe_required(model_.files())) {
      runtime_probe_ = reco::gui::probe_gui_runtime();
    }
    preview_readiness_ = reco::gui::evaluate_preview_runtime(model_.files(), runtime_probe_);
    model_.set_gpu_ready(preview_readiness_.ready);
  }

  void refresh() {
    const auto status = model_.preview_status();
    const auto status_name = QString::fromUtf8(
        preview_status_name(status).data(), static_cast<int>(preview_status_name(status).size()));
    const auto runtime_name = reco::gui::preview_runtime_state_name(preview_readiness_.state);
    const QString runtime_status =
        QString::fromUtf8(runtime_name.data(), static_cast<int>(runtime_name.size()));
    status_->setText("GPU preview status: " + status_name + "\nRuntime: " + runtime_status +
                     "\n" + QString::fromStdString(preview_readiness_.detail));
    export_->setEnabled(model_.can_export());
    statusBar()->showMessage("GPU preview: " + status_name);
  }

  reco::gui::GuiAppModel model_{reco::gui::GuiSettings::load()};
  reco::gui::GuiRuntimeProbe runtime_probe_;
  reco::gui::PreviewRuntimeReadiness preview_readiness_;
  QLineEdit* left_ = nullptr;
  QLineEdit* right_ = nullptr;
  QLineEdit* calibration_ = nullptr;
  QSlider* fov_ = nullptr;
  QDoubleSpinBox* blend_ = nullptr;
  QCheckBox* play_ = nullptr;
  QComboBox* codec_ = nullptr;
  QComboBox* quality_ = nullptr;
  QPushButton* export_ = nullptr;
  QFrame* preview_surface_ = nullptr;
  QLabel* status_ = nullptr;
};

} // namespace

int main(int argc, char** argv) {
  QApplication app(argc, argv);
  MainWindow window;
  window.show();
  return QApplication::exec();
}
