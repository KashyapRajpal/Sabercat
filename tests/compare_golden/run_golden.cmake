# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

# Golden-output test for sabercat_compare. Each case runs the exe on fixed
# fixtures and records stdout, stderr, the exit code and any map PNG it writes.
# MODE=capture stores those in EXPECTED from a known-good exe. MODE=check
# compares them byte-for-byte and fails, naming the case and file, on any
# difference. A missing expected file is a failure. A PNG passes only when the
# bytes match and the decoded pixels match: PIXEL_TOOL (sabercat_compare_png_pixels)
# compares the pixels, and its result is printed for each PNG.
#
#   cmake -DEXE=<exe> -DFIXTURES=<dir> -DFLIP_IMAGES=<dir> -DEXPECTED=<dir>
#         -DWORK=<dir> -DPIXEL_TOOL=<exe> -DMODE=check|capture -P run_golden.cmake

foreach(var EXE FIXTURES FLIP_IMAGES EXPECTED WORK MODE)
    if(NOT DEFINED ${var} OR "${${var}}" STREQUAL "")
        message(FATAL_ERROR "run_golden.cmake: -D${var}=... is required")
    endif()
endforeach()
if(NOT MODE STREQUAL "check" AND NOT MODE STREQUAL "capture")
    message(FATAL_ERROR "run_golden.cmake: MODE must be check or capture, got '${MODE}'")
endif()
if(MODE STREQUAL "check")
    if(NOT DEFINED PIXEL_TOOL OR "${PIXEL_TOOL}" STREQUAL "" OR NOT EXISTS "${PIXEL_TOOL}")
        message(FATAL_ERROR "run_golden.cmake: the pixel comparison tool is missing: ${PIXEL_TOOL} (build the sabercat_compare_png_pixels target)")
    endif()
endif()

# One case per line, as "name|arguments". Arguments are split on spaces, and
# file names are relative to WORK. Output maps named by --flip-map or
# --diff-map are recorded too. To add a case, add one line here.
set(GOLDEN_CASES
    # The help and usage-error expected files (help.out, help_with_args.out, and the
    # .err files for unknown_flag, ppd_zero, ppd_big, ppd_abc, ppd_missing_value,
    # flip_map_missing_value, diff_map_missing_value, noflip_flipmap and ppd_nan) are
    # the 1.0.1 detail lines, with a new four-line synopsis that lists every flag. The
    # no-argument case and the one- and three-positional cases are not here because
    # they now open the GUI.
    "identical|a.png a.png"
    "different|a.png b.png"
    "maps|a.png b.png --flip-map flip.png --diff-map diff.png"
    "noflip|a.png b.png --no-flip --diff-map diff.png"
    "ppd30|a.png b.png --ppd 30"
    "ppd_float|a.png b.png --ppd 67.5"
    "size_mismatch|a.png c.png"
    "missing_file|missing.png a.png"
    "bad_png|bad.png a.png"
    "flip_pair|ref_flip.png test_flip.png --flip-map flipmap.png --diff-map diffmap.png"
    "flip_pair_ppd|ref_flip.png test_flip.png --ppd 40"
    "help|--help"
    "help_with_args|a.png b.png --help"
    "unknown_flag|a.png b.png --bogus"
    "ppd_zero|a.png b.png --ppd 0"
    "ppd_big|a.png b.png --ppd 1001"
    "ppd_abc|a.png b.png --ppd abc"
    "ppd_missing_value|a.png b.png --ppd"
    "flip_map_missing_value|a.png b.png --flip-map"
    "diff_map_missing_value|a.png b.png --diff-map"
    "noflip_flipmap|a.png b.png --no-flip --flip-map f.png"
    "unwritable_map|a.png b.png --diff-map no_such_dir/x.png"
    "ppd_nan|a.png b.png --ppd nan"
)

# Fresh work directory: the fixtures, then the FLIP pair under the names the
# flip_pair cases use.
foreach(flip_image reference.png test.png)
    if(NOT EXISTS "${FLIP_IMAGES}/${flip_image}")
        message(FATAL_ERROR "run_golden.cmake: missing FLIP image ${FLIP_IMAGES}/${flip_image}")
    endif()
endforeach()
file(REMOVE_RECURSE "${WORK}")
file(MAKE_DIRECTORY "${WORK}")
file(COPY "${FIXTURES}/" DESTINATION "${WORK}")
file(COPY_FILE "${FLIP_IMAGES}/reference.png" "${WORK}/ref_flip.png")
file(COPY_FILE "${FLIP_IMAGES}/test.png" "${WORK}/test_flip.png")

if(MODE STREQUAL "capture")
    file(MAKE_DIRECTORY "${EXPECTED}")
endif()

# Handles one recorded file. In capture mode it is copied to EXPECTED, or a
# stale expected file is removed when the case wrote nothing. In check mode the
# source and expected files must both exist, or both be absent, and match
# byte-for-byte. Failures are appended to `failures` in the caller's scope.
macro(golden_record source_file expected_file leaf)
    if(MODE STREQUAL "capture")
        if(EXISTS "${source_file}")
            file(COPY_FILE "${source_file}" "${expected_file}")
        else()
            file(REMOVE "${expected_file}")
        endif()
    else()
        if(EXISTS "${source_file}" AND NOT EXISTS "${expected_file}")
            list(APPEND failures "${name}: ${leaf} was written but has no expected file")
        elseif(NOT EXISTS "${source_file}" AND EXISTS "${expected_file}")
            list(APPEND failures "${name}: ${leaf} was expected but not written")
        elseif(EXISTS "${source_file}")
            execute_process(
                COMMAND "${CMAKE_COMMAND}" -E compare_files "${source_file}" "${expected_file}"
                RESULT_VARIABLE golden_differs
                OUTPUT_QUIET
                ERROR_QUIET
            )
            if("${leaf}" MATCHES "\\.png$")
                # The decoded pixels are compared too. The status line is printed for
                # every PNG, whether it passes or fails.
                execute_process(
                    COMMAND "${PIXEL_TOOL}" "${source_file}" "${expected_file}"
                    RESULT_VARIABLE pixel_status
                    OUTPUT_VARIABLE pixel_text
                    ERROR_VARIABLE pixel_error
                )
                string(STRIP "${pixel_text}" pixel_text)
                string(STRIP "${pixel_error}" pixel_error)
                set(pixel_line "${pixel_text}")
                if(NOT "${pixel_error}" STREQUAL "")
                    string(APPEND pixel_line " (stderr: ${pixel_error})")
                endif()
                message(STATUS "golden pixels ${leaf}: ${pixel_line} (exit ${pixel_status})")

                set(bytes_differ FALSE)
                set(pixels_differ FALSE)
                if(NOT golden_differs EQUAL 0)
                    set(bytes_differ TRUE)
                endif()
                if(NOT pixel_status EQUAL 0)
                    set(pixels_differ TRUE)
                endif()
                if(pixels_differ)
                    list(APPEND failures "${name}: ${leaf} differs from ${expected_file} (pixels differ: ${pixel_line})")
                elseif(bytes_differ)
                    message(STATUS "pixels identical, bytes differ: ${leaf}")
                    list(APPEND failures "${name}: ${leaf} differs from ${expected_file} (bytes only; decoded pixels are identical)")
                endif()
            elseif(NOT golden_differs EQUAL 0)
                list(APPEND failures "${name}: ${leaf} differs from ${expected_file}")
            endif()
        endif()
    endif()
endmacro()

set(failures "")
set(case_count 0)

foreach(entry IN LISTS GOLDEN_CASES)
    string(REPLACE "|" ";" parts "${entry}")
    list(GET parts 0 name)
    list(GET parts 1 argument_text)
    separate_arguments(args UNIX_COMMAND "${argument_text}")
    math(EXPR case_count "${case_count} + 1")

    # Map outputs are the value after each --flip-map or --diff-map.
    set(outputs "")
    set(previous "")
    foreach(token IN LISTS args)
        if("${previous}" STREQUAL "--flip-map" OR "${previous}" STREQUAL "--diff-map")
            list(APPEND outputs "${token}")
        endif()
        set(previous "${token}")
    endforeach()

    # Remove stale outputs so a case never sees an earlier case's file.
    foreach(output IN LISTS outputs)
        file(REMOVE "${WORK}/${output}")
    endforeach()

    execute_process(
        COMMAND "${EXE}" ${args}
        WORKING_DIRECTORY "${WORK}"
        OUTPUT_FILE "${WORK}/${name}.out"
        ERROR_FILE "${WORK}/${name}.err"
        RESULT_VARIABLE exit_code
    )
    file(WRITE "${WORK}/${name}.code" "${exit_code}\n")

    golden_record("${WORK}/${name}.out" "${EXPECTED}/${name}.out" "${name}.out")
    golden_record("${WORK}/${name}.err" "${EXPECTED}/${name}.err" "${name}.err")
    golden_record("${WORK}/${name}.code" "${EXPECTED}/${name}.code" "${name}.code")
    foreach(output IN LISTS outputs)
        string(REPLACE "/" "_" flat "${output}")
        golden_record("${WORK}/${output}" "${EXPECTED}/${name}.${flat}" "${name}.${flat}")
    endforeach()
endforeach()

if(MODE STREQUAL "capture")
    message(STATUS "run_golden.cmake: captured ${case_count} cases into ${EXPECTED}")
elseif(failures)
    list(JOIN failures "\n  " failure_text)
    message(FATAL_ERROR "run_golden.cmake: golden output mismatch:\n  ${failure_text}")
else()
    message(STATUS "run_golden.cmake: ${case_count} golden cases match")
endif()
