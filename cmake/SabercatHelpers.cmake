# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

# Helpers for projects that consume Sabercat, through either
# find_package(Sabercat) or add_subdirectory(Sabercat). Included automatically
# by both paths.

# Copies every runtime DLL `target` depends on -- sabercat_c.dll and the
# vendored zlib DLL that libsabercat's .spz loader needs -- beside the
# target's output after each build. Windows has no rpath, so an executable
# that links Sabercat::sabercat or Sabercat::sabercat_c dies before main()
# without them. A no-op where nothing needs copying.
function(sabercat_copy_runtime_dlls target)
    set(_dlls "$<TARGET_RUNTIME_DLLS:${target}>")
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND "$<$<BOOL:${_dlls}>:${CMAKE_COMMAND};-E;copy_if_different;${_dlls};$<TARGET_FILE_DIR:${target}>>"
        COMMAND_EXPAND_LISTS
        VERBATIM
    )
endfunction()
