def _split_flags(raw):
    return [flag for flag in raw.strip().split(" ") if flag]

def _partition_cflags(flags):
    includes = []
    defines = []
    copts = []
    for flag in flags:
        if flag.startswith("-I"):
            includes.append(flag[2:])
        elif flag.startswith("-D"):
            defines.append(flag[2:])
        else:
            copts.append(flag)
    return includes, defines, copts

def _qt_configure_impl(repository_ctx):
    pkg_config = repository_ctx.which("pkg-config")
    if not pkg_config:
        fail("pkg-config is required to configure Qt6 Widgets for Bazel")

    cflags = repository_ctx.execute([pkg_config, "--cflags", "Qt6Widgets"])
    if cflags.return_code != 0:
        fail("pkg-config --cflags Qt6Widgets failed: {}".format(cflags.stderr))

    libs = repository_ctx.execute([pkg_config, "--libs", "Qt6Widgets"])
    if libs.return_code != 0:
        fail("pkg-config --libs Qt6Widgets failed: {}".format(libs.stderr))

    include_paths, defines, copts = _partition_cflags(_split_flags(cflags.stdout))
    relative_includes = []
    for i, include_path in enumerate(include_paths):
        link_name = "include{}".format(i)
        repository_ctx.symlink(include_path, link_name)
        relative_includes.append(link_name)

    repository_ctx.file("BUILD.bazel", """\
load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "qt_widgets",
    copts = {copts},
    defines = {defines},
    includes = {includes},
    hdrs = glob(["include*/**"]),
    linkopts = {linkopts},
    visibility = ["//visibility:public"],
)
""".format(
        copts = repr(copts),
        defines = repr(defines),
        includes = repr(relative_includes),
        linkopts = repr(_split_flags(libs.stdout)),
    ))

qt_configure = repository_rule(
    implementation = _qt_configure_impl,
    local = True,
)
