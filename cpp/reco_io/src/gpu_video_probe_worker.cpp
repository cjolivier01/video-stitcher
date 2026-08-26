#include "gpu_video_probe_internal.hpp"
#include "gpu_video_probe_protocol.hpp"

#include <array>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace reco::io::detail {
namespace {

#if defined(_WIN32)
class StartupAttributes {
public:
  StartupAttributes() {
    SIZE_T size = 0;
    (void)InitializeProcThreadAttributeList(nullptr, 1, 0, &size);
    if (size == 0) {
      return;
    }
    storage_.resize(size);
    value_ = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(storage_.data());
    if (InitializeProcThreadAttributeList(value_, 1, 0, &size) == 0) {
      value_ = nullptr;
    }
  }
  StartupAttributes(const StartupAttributes&) = delete;
  StartupAttributes& operator=(const StartupAttributes&) = delete;
  ~StartupAttributes() {
    if (value_ != nullptr) {
      DeleteProcThreadAttributeList(value_);
    }
  }

  [[nodiscard]] LPPROC_THREAD_ATTRIBUTE_LIST get() const { return value_; }

private:
  std::vector<unsigned char> storage_;
  LPPROC_THREAD_ATTRIBUTE_LIST value_ = nullptr;
};
#endif

std::string read_request() {
  const auto read_exact = [](char* destination, std::size_t size) {
    std::cin.read(destination, static_cast<std::streamsize>(size));
    if (std::cin.gcount() != static_cast<std::streamsize>(size)) {
      throw GpuVideoProbeError("video probe worker request has a truncated IPC frame");
    }
  };
  ProbeIpcFrameHeader header{};
  read_exact(header.data(), header.size());
  std::string request(decode_probe_ipc_frame_header(header), '\0');
  read_exact(request.data(), request.size());
  return request;
}

void write_response(std::string_view response) {
  const auto header = encode_probe_ipc_frame_header(response.size());
  std::cout.write(header.data(), static_cast<std::streamsize>(header.size()));
  std::cout.write(response.data(), static_cast<std::streamsize>(response.size()));
  std::cout.flush();
  if (!std::cout) {
    throw GpuVideoProbeError("failed to write video probe worker response");
  }
}

} // namespace

#if defined(_WIN32)
int run_gpu_video_probe_guardian() {
  std::vector<wchar_t> executable(32'768);
  const auto length =
      GetModuleFileNameW(nullptr, executable.data(), static_cast<DWORD>(executable.size()));
  if (length == 0 || length >= executable.size()) {
    return 2;
  }
  const std::wstring application(executable.data(), length);
  auto command_line = L"\"" + application + L"\" --reco-video-probe-worker";
  STARTUPINFOEXW startup{};
  startup.StartupInfo.cb = sizeof(startup);
  startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
  startup.StartupInfo.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
  startup.StartupInfo.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
  startup.StartupInfo.hStdError = GetStdHandle(STD_ERROR_HANDLE);
  std::array<HANDLE, 3> inherited_handles{
      startup.StartupInfo.hStdInput, startup.StartupInfo.hStdOutput, startup.StartupInfo.hStdError};
  StartupAttributes attributes;
  if (attributes.get() == nullptr ||
      UpdateProcThreadAttribute(attributes.get(), 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                                inherited_handles.data(), sizeof(inherited_handles), nullptr,
                                nullptr) == 0) {
    return 2;
  }
  startup.lpAttributeList = attributes.get();
  PROCESS_INFORMATION process{};
  if (CreateProcessW(application.c_str(), command_line.data(), nullptr, nullptr, TRUE,
                     CREATE_NO_WINDOW | CREATE_SUSPENDED | EXTENDED_STARTUPINFO_PRESENT, nullptr,
                     nullptr, &startup.StartupInfo, &process) == 0) {
    return 2;
  }
  BOOL worker_in_job = FALSE;
  const bool contained =
      IsProcessInJob(process.hProcess, nullptr, &worker_in_job) != 0 && worker_in_job != FALSE;
  if (!contained || ResumeThread(process.hThread) == std::numeric_limits<DWORD>::max()) {
    (void)TerminateProcess(process.hProcess, 2);
    (void)CloseHandle(process.hThread);
    (void)CloseHandle(process.hProcess);
    return 2;
  }
  (void)CloseHandle(process.hThread);
  const auto wait_result = WaitForSingleObject(process.hProcess, INFINITE);
  DWORD exit_code = 2;
  if (wait_result != WAIT_OBJECT_0 || GetExitCodeProcess(process.hProcess, &exit_code) == 0) {
    exit_code = 2;
  }
  (void)CloseHandle(process.hProcess);
  return exit_code <= static_cast<DWORD>(std::numeric_limits<int>::max())
             ? static_cast<int>(exit_code)
             : 2;
}
#endif

int run_gpu_video_probe_worker() {
  std::string response;
  try {
    const auto payload = read_request();
    const auto request = decode_probe_request(payload);
    response = encode_probe_success(probe_gpu_video_in_process(request.config, request.timeout_ns));
  } catch (const std::invalid_argument& error) {
    response = encode_probe_failure("invalid_argument", error.what());
  } catch (const GpuVideoProbeError& error) {
    response = encode_probe_failure("probe_error", error.what());
  } catch (const std::exception& error) {
    response = encode_probe_failure("worker_error", error.what());
  } catch (...) {
    response = encode_probe_failure("worker_error", "unknown video probe worker failure");
  }
  try {
    write_response(response);
    return 0;
  } catch (...) {
    return 2;
  }
}

} // namespace reco::io::detail
