# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

# CLI checks for sabercat_compare's HDR, report and launch-mode behavior. Each case
# runs the exe and checks the exit code, and where stated the output lines, the
# files it writes and the stderr text. All failures are collected and reported
# together at the end. The PNG pair a.png and b.png comes from this folder's
# fixtures/ subfolder.
#
#   cmake -DEXE=<exe> -DFLIP_IMAGES=<dir> -DWORK=<dir> -P run_cli_checks.cmake

foreach(var EXE FLIP_IMAGES WORK)
    if(NOT DEFINED ${var} OR "${${var}}" STREQUAL "")
        message(FATAL_ERROR "run_cli_checks.cmake: -D${var}=... is required")
    endif()
endforeach()
foreach(flip_image reference.exr test.exr reference.png test.png)
    if(NOT EXISTS "${FLIP_IMAGES}/${flip_image}")
        message(FATAL_ERROR "run_cli_checks.cmake: missing FLIP image ${FLIP_IMAGES}/${flip_image}")
    endif()
endforeach()

file(REMOVE_RECURSE "${WORK}")
file(MAKE_DIRECTORY "${WORK}")
foreach(flip_image reference.exr test.exr reference.png test.png)
    file(COPY_FILE "${FLIP_IMAGES}/${flip_image}" "${WORK}/${flip_image}")
endforeach()
foreach(fixture a.png b.png)
    if(NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/fixtures/${fixture}")
        message(FATAL_ERROR "run_cli_checks.cmake: missing fixture ${CMAKE_CURRENT_LIST_DIR}/fixtures/${fixture}")
    endif()
    file(COPY_FILE "${CMAKE_CURRENT_LIST_DIR}/fixtures/${fixture}" "${WORK}/${fixture}")
endforeach()

set(failures "")
set(case_count 0)

# Runs the exe with the arguments after the two leading ones, in WORK. Sets
# cli_code, cli_out and cli_err in the caller's scope.
macro(run_cli)
    math(EXPR case_count "${case_count} + 1")
    execute_process(
        COMMAND "${EXE}" ${ARGN}
        WORKING_DIRECTORY "${WORK}"
        OUTPUT_VARIABLE cli_out
        ERROR_VARIABLE cli_err
        RESULT_VARIABLE cli_code
    )
endmacro()

# Checks the exit code of the last run_cli(): name, then the expected code.
macro(expect_code name expected)
    if(NOT "${cli_code}" STREQUAL "${expected}")
        list(APPEND failures "${name}: exit code ${cli_code}, expected ${expected}\n${cli_err}")
    endif()
endmacro()

# Checks that the last run_cli() output (stdout or stderr) matches a regex. This is a
# function, not a macro: a macro re-evaluates its arguments, which turns the escaped
# regex metacharacters (\( \. ...) back into plain ones and breaks the match.
function(expect_match name stream regex)
    if(NOT "${${stream}}" MATCHES "${regex}")
        list(APPEND failures "${name}: ${stream} does not match '${regex}'")
        set(failures "${failures}" PARENT_SCOPE)
    endif()
endfunction()

# Checks that a file exists and is not empty.
macro(expect_file name path)
    if(NOT EXISTS "${path}")
        list(APPEND failures "${name}: ${path} was not written")
    else()
        file(SIZE "${path}" file_size)
        if(file_size EQUAL 0)
            list(APPEND failures "${name}: ${path} is empty")
        endif()
    endif()
endmacro()

# CMake has no floating-point arithmetic, so the report's JSON numbers are compared with
# the printed text by rounding both with decimal digit strings. A decimal is held as
# "digits|point": its significant digits with no leading or trailing zeros, and the
# value 0.<digits> * 10^<point>. Zero is "|0".

# Converts printf-style decimal text ([digits][.digits][e[+-]digits]) to "digits|point".
function(cli_decimal text out)
    if(NOT "${text}" MATCHES "^([0-9]*)\\.?([0-9]*)([eE]([-+]?[0-9]+))?$")
        message(FATAL_ERROR "run_cli_checks.cmake: cannot read the number '${text}'")
    endif()
    set(int_part "${CMAKE_MATCH_1}")
    set(frac_part "${CMAKE_MATCH_2}")
    set(exp_text "${CMAKE_MATCH_4}")
    set(exponent 0)
    if(NOT "${exp_text}" STREQUAL "")
        string(REGEX REPLACE "^\\+" "" exp_text "${exp_text}")
        string(REGEX REPLACE "^(-?)0+([0-9])" "\\1\\2" exp_text "${exp_text}")
        math(EXPR exponent "${exp_text}")
    endif()
    string(LENGTH "${int_part}" int_len)
    set(all_digits "${int_part}${frac_part}")
    string(LENGTH "${all_digits}" all_len)
    string(REGEX REPLACE "^0+" "" digits "${all_digits}")
    string(LENGTH "${digits}" digits_len)
    if("${digits}" STREQUAL "")
        set(${out} "|0" PARENT_SCOPE)
        return()
    endif()
    math(EXPR point "${int_len} + ${exponent} - (${all_len} - ${digits_len})")
    set(${out} "${digits}|${point}" PARENT_SCOPE)
endfunction()

# Rounds a "digits|point" form the way printf does: keep is how many leading digits stay
# (a negative keep gives zero). Exact ties go to the even digit. The rounding works on the
# JSON's shortest decimal text, so a value whose text ends in a 5 right at the printed
# position can round differently from printf's exact binary value. Computed metrics do not
# land there in practice. Trailing zeros are dropped from the result.
function(cli_round_digits form keep out)
    if(NOT "${form}" MATCHES "^([0-9]*)\\|(-?[0-9]+)$")
        message(FATAL_ERROR "run_cli_checks.cmake: bad decimal form '${form}'")
    endif()
    set(kept "${CMAKE_MATCH_1}")
    set(point "${CMAKE_MATCH_2}")
    string(LENGTH "${kept}" kept_len)
    if("${kept}" STREQUAL "" OR "${keep}" LESS 0)
        set(${out} "|0" PARENT_SCOPE)
        return()
    endif()
    if("${keep}" LESS "${kept_len}")
        # The digits from keep on decide the rounding. An exact tie (a 5 and nothing
        # after it) goes to the even digit, as printf does.
        string(SUBSTRING "${kept}" ${keep} -1 tail)
        string(SUBSTRING "${tail}" 0 1 next)
        set(round_up FALSE)
        if("${next}" GREATER 5)
            set(round_up TRUE)
        elseif("${next}" EQUAL 5)
            if(NOT "${tail}" MATCHES "^50*$")
                set(round_up TRUE)
            elseif("${keep}" GREATER 0)
                math(EXPR last_index "${keep} - 1")
                string(SUBSTRING "${kept}" ${last_index} 1 last)
                math(EXPR last_odd "${last} % 2")
                if("${last_odd}" EQUAL 1)
                    set(round_up TRUE)
                endif()
            endif()
        endif()
        string(SUBSTRING "${kept}" 0 ${keep} head)
        if(round_up)
            if("${keep}" EQUAL 0)
                set(head "1")
                math(EXPR point "${point} + 1")
            else()
                math(EXPR head "${head} + 1")
                string(LENGTH "${head}" head_len)
                if("${head_len}" GREATER "${keep}")
                    set(head "1")
                    math(EXPR point "${point} + 1")
                endif()
            endif()
        endif()
        set(kept "${head}")
    endif()
    string(REGEX REPLACE "0+$" "" kept "${kept}")
    if("${kept}" STREQUAL "")
        set(${out} "|0" PARENT_SCOPE)
    else()
        set(${out} "${kept}|${point}" PARENT_SCOPE)
    endif()
endfunction()

# The number as printf would show it at the given precision, in "digits|point" form:
# mode fixed keeps count digits after the point (%.<count>f), mode sig keeps count
# significant digits (%.<count>g). Pass sig with a large count to canonicalize text.
function(cli_rounded text mode count out)
    cli_decimal("${text}" form)
    if(NOT "${form}" MATCHES "^[0-9]*\\|(-?[0-9]+)$")
        message(FATAL_ERROR "run_cli_checks.cmake: bad decimal form '${form}'")
    endif()
    set(point "${CMAKE_MATCH_1}")
    if("${mode}" STREQUAL "fixed")
        math(EXPR keep "${point} + ${count}")
    else()
        set(keep "${count}")
    endif()
    cli_round_digits("${form}" "${keep}" rounded)
    set(${out} "${rounded}" PARENT_SCOPE)
endfunction()

# Help and the window-only usage errors.
run_cli(--help)
expect_code("help" 0)
expect_match("help" cli_out "usage: sabercat_compare")

# A launch with no arguments opens the window, which a CPU-only check cannot do; the
# window-only arguments are checked for their usage errors instead.
run_cli(--selftest)
expect_code("selftest without a directory" 2)
expect_match("selftest without a directory" cli_err "--selftest needs a directory")

# HDR on the EXR pair: the FLIP line, the exposure map and the report.
run_cli(reference.exr test.exr --hdr --exposure-map exposure.png --report hdr.json)
expect_code("hdr exr pair" 0)
expect_match("hdr exr pair" cli_out "FLIP \\(HDR, aces\\): 0\\.2835")
expect_match("hdr exr pair" cli_out "linear MSE: ")
expect_match("hdr exr pair" cli_out "PSNR: ")
expect_file("hdr exr pair" "${WORK}/exposure.png")
expect_file("hdr exr pair" "${WORK}/hdr.json")

# The same report twice is byte-identical.
run_cli(reference.exr test.exr --hdr --report hdr_again.json)
expect_code("report repeat" 0)
execute_process(COMMAND "${CMAKE_COMMAND}" -E compare_files "${WORK}/hdr.json" "${WORK}/hdr_again.json"
    RESULT_VARIABLE report_differs OUTPUT_QUIET ERROR_QUIET)
if(NOT report_differs EQUAL 0)
    list(APPEND failures "report repeat: the two reports differ")
endif()

# A CSV report: the header row comes first.
run_cli(reference.exr test.exr --hdr --report hdr.CSV)
expect_code("csv report" 0)
expect_file("csv report" "${WORK}/hdr.CSV")
file(STRINGS "${WORK}/hdr.CSV" csv_lines LIMIT_COUNT 1)
if(NOT csv_lines MATCHES "^mode,reference_path,")
    list(APPEND failures "csv report: the header row is '${csv_lines}'")
endif()

# An LDR run on EXR input prints the display-transform note on stderr.
run_cli(reference.exr test.exr)
expect_code("exr without hdr" 0)
expect_match("exr without hdr" cli_err "note: .*reference\\.exr is linear \\(EXR\\); compared after the sRGB display transform")

# HDR on PNG input: the warning on stderr, and success.
run_cli(reference.png test.png --hdr)
expect_code("hdr on png" 0)
expect_match("hdr on png" cli_err "warning: --hdr with display-encoded input 'reference.png'")

# Invalid combinations are usage errors (exit 2).
run_cli(reference.exr test.exr --tone-mapper aces)
expect_code("tone mapper without hdr" 2)
run_cli(reference.exr test.exr --exposure-map exposure.png)
expect_code("exposure map without hdr" 2)
run_cli(reference.exr test.exr --hdr --no-flip)
expect_code("hdr with no-flip" 2)
run_cli(reference.exr test.exr --hdr --start-exposure 1 --stop-exposure 0)
expect_code("start above stop" 2)
run_cli(reference.exr test.exr --hdr --num-exposures 1)
expect_code("one exposure" 2)
run_cli(reference.exr test.exr --hdr --num-exposures abc)
expect_code("non-integer exposure count" 2)
run_cli(reference.exr test.exr --hdr --tone-mapper foo)
expect_code("unknown tone mapper" 2)
expect_match("unknown tone mapper" cli_err "aces, hable or reinhard")
run_cli(reference.exr test.exr --hdr --start-exposure nan)
expect_code("nan exposure" 2)

# An unwritable report path is a failure (exit 1).
run_cli(reference.exr test.exr --hdr --report no_such_dir/report.json)
expect_code("unwritable report" 1)

# Two PNGs without --report take the loadPng path; with --report they take loadImage.
# The numbers must be the same, and the JSON report must agree with the printed lines
# to the printed precision (PSNR %.2f, MSE %.6g, FLIP %.6f).
run_cli(a.png b.png)
expect_code("pair without report" 0)
set(plain_stdout "${cli_out}")
run_cli(a.png b.png --report pair_metrics.json)
expect_code("pair with report" 0)
set(report_stdout "${cli_out}")
if(NOT "${plain_stdout}" STREQUAL "${report_stdout}")
    list(APPEND failures "pair with report: stdout differs from the run without --report\n${plain_stdout}--- with report:\n${report_stdout}")
endif()

set(stdout_psnr "")
set(stdout_mse "")
set(stdout_flip "")
if("${report_stdout}" MATCHES "PSNR: ([-+.0-9eE]+) dB")
    set(stdout_psnr "${CMAKE_MATCH_1}")
endif()
if("${report_stdout}" MATCHES "MSE: ([-+.0-9eE]+)")
    set(stdout_mse "${CMAKE_MATCH_1}")
endif()
if("${report_stdout}" MATCHES "FLIP: ([-+.0-9eE]+) \\(ppd")
    set(stdout_flip "${CMAKE_MATCH_1}")
endif()
if("${stdout_psnr}" STREQUAL "" OR "${stdout_mse}" STREQUAL "" OR "${stdout_flip}" STREQUAL "")
    list(APPEND failures "pair with report: PSNR, MSE or FLIP line missing from stdout:\n${report_stdout}")
endif()

expect_file("pair with report" "${WORK}/pair_metrics.json")
if(EXISTS "${WORK}/pair_metrics.json")
    file(READ "${WORK}/pair_metrics.json" pair_json)
    string(JSON json_psnr ERROR_VARIABLE json_error GET "${pair_json}" metrics psnrDb)
    if(NOT "${json_error}" STREQUAL "NOTFOUND")
        list(APPEND failures "pair with report: psnrDb missing: ${json_error}")
    else()
        cli_rounded("${json_psnr}" fixed 2 json_form)
        cli_rounded("${stdout_psnr}" sig 99 stdout_form)
        if(NOT "${json_form}" STREQUAL "${stdout_form}")
            list(APPEND failures "pair with report: psnrDb ${json_psnr} does not match the PSNR line '${stdout_psnr}' (PSNR %.2f)")
        endif()
    endif()

    string(JSON json_mse ERROR_VARIABLE json_error GET "${pair_json}" metrics mse)
    if(NOT "${json_error}" STREQUAL "NOTFOUND")
        list(APPEND failures "pair with report: mse missing: ${json_error}")
    else()
        cli_rounded("${json_mse}" sig 6 json_form)
        cli_rounded("${stdout_mse}" sig 99 stdout_form)
        if(NOT "${json_form}" STREQUAL "${stdout_form}")
            list(APPEND failures "pair with report: mse ${json_mse} does not match the MSE line '${stdout_mse}' (MSE %.6g)")
        endif()
    endif()

    string(JSON json_flip ERROR_VARIABLE json_error GET "${pair_json}" metrics flipMean)
    if(NOT "${json_error}" STREQUAL "NOTFOUND")
        list(APPEND failures "pair with report: flipMean missing: ${json_error}")
    else()
        cli_rounded("${json_flip}" fixed 6 json_form)
        cli_rounded("${stdout_flip}" sig 99 stdout_form)
        if(NOT "${json_form}" STREQUAL "${stdout_form}")
            list(APPEND failures "pair with report: flipMean ${json_flip} does not match the FLIP line '${stdout_flip}' (FLIP %.6f)")
        endif()
    endif()
endif()

list(LENGTH failures failure_count)
if(failure_count GREATER 0)
    list(JOIN failures "\n  " failure_text)
    message(FATAL_ERROR "run_cli_checks.cmake: ${failure_count} of ${case_count} CLI checks failed:\n  ${failure_text}")
else()
    message(STATUS "run_cli_checks.cmake: ${case_count} CLI runs match")
endif()
