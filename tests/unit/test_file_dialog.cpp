// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// The pure parts of the native file dialogs (apps/common/file_dialog.h): the
// filter specs the dialogs take, the default-extension rule and the UTF-8 /
// UTF-16 conversions. No dialog is shown. Non-ASCII text is written as byte
// escapes, so the source file stays ASCII; each escape sits in its own literal
// so that a following hex-digit letter cannot extend it.
//
// AsyncFileDialog is tested with an injected runner that blocks on a latch the
// test controls, so its state machine runs without any dialog.
#include "unit_test.h"

#include "file_dialog.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>

namespace sabercat::unit
{

namespace
{

using app::FileFilter;

// "Ünïcödé 層/проект" -- the name used by the UTF-8 tests.
const std::string kUnicodeName = "\xC3\x9C"
                                 "n"
                                 "\xC3\xAF"
                                 "c"
                                 "\xC3\xB6"
                                 "d"
                                 "\xC3\xA9"
                                 " "
                                 "\xE5\xB1\xA4"
                                 "/"
                                 "\xD0\xBF"
                                 "\xD1\x80"
                                 "\xD0\xBE"
                                 "\xD0\xB5"
                                 "\xD0\xBA"
                                 "\xD1\x82";

void testDefaultExtension(Context & ctx)
{
    // A name without an extension takes the default, with or without its dot.
    UNIT_CHECK(ctx, app::withDefaultExtension("scene", "sabercat") == "scene.sabercat");
    UNIT_CHECK(ctx, app::withDefaultExtension("scene", ".sabercat") == "scene.sabercat");
    // A name that already has one keeps it, whatever it is.
    UNIT_CHECK(ctx, app::withDefaultExtension("scene.sabercat", "sabercat") == "scene.sabercat");
    UNIT_CHECK(ctx, app::withDefaultExtension("scene.ply", "sabercat") == "scene.ply");
    UNIT_CHECK(ctx, app::withDefaultExtension("archive.tar.gz", "sabercat") == "archive.tar.gz");
    // A trailing dot is no extension: it is replaced by the default.
    UNIT_CHECK(ctx, app::withDefaultExtension("scene.", "sabercat") == "scene.sabercat");
    // A leading dot is not an extension either.
    UNIT_CHECK(ctx, app::withDefaultExtension(".hidden", "sabercat") == ".hidden.sabercat");
    // Empty name or empty extension: nothing to do.
    UNIT_CHECK(ctx, app::withDefaultExtension("", "sabercat") == "");
    UNIT_CHECK(ctx, app::withDefaultExtension("scene", "") == "scene");
    UNIT_CHECK(ctx, app::withDefaultExtension("scene", ".") == "scene");
    // A dot in a directory name is not the file's extension.
    UNIT_CHECK(ctx, app::withDefaultExtension("D:\\my.dir\\scene", "sabercat")
                        == "D:\\my.dir\\scene.sabercat");
    UNIT_CHECK(ctx, app::withDefaultExtension("my.dir/scene", "sabercat") == "my.dir/scene.sabercat");
    UNIT_CHECK(ctx, app::withDefaultExtension("my.dir/scene.ply", "sabercat") == "my.dir/scene.ply");
    // Non-ASCII names: the separator and dot are ASCII, so the bytes split cleanly.
    UNIT_CHECK(ctx, app::withDefaultExtension(kUnicodeName + ".sabercat", "sabercat")
                        == kUnicodeName + ".sabercat");
    UNIT_CHECK(ctx, app::withDefaultExtension(kUnicodeName, "sabercat") == kUnicodeName + ".sabercat");
    UNIT_CHECK(ctx, app::withDefaultExtension(kUnicodeName + ".", "sabercat")
                        == kUnicodeName + ".sabercat");
}

void testUtf8Wide(Context & ctx)
{
    // Empty strings stay empty in both directions.
    UNIT_CHECK(ctx, app::utf8ToWide("").empty());
    UNIT_CHECK(ctx, app::wideToUtf8(std::wstring()).empty());
    UNIT_CHECK(ctx, app::utf8ToWide("scene.sabercat") == L"scene.sabercat");
    // Known code points: U+00DC, U+5C64 (a CJK ideograph) and an emoji outside
    // the BMP, which needs a surrogate pair in UTF-16.
    UNIT_CHECK(ctx, app::utf8ToWide("\xC3\x9C") == L"\u00DC");
    UNIT_CHECK(ctx, app::utf8ToWide("\xE5\xB1\xA4") == L"\u5C64");
    UNIT_CHECK(ctx, app::utf8ToWide("\xF0\x9F\x98\x80") == L"\U0001F600");
    // Round trips.
    const std::string sample = kUnicodeName + ".sabercat";
    UNIT_CHECK(ctx, app::wideToUtf8(app::utf8ToWide(sample)) == sample);
    UNIT_CHECK(ctx, app::wideToUtf8(app::utf8ToWide("\xF0\x9F\x98\x80")) == "\xF0\x9F\x98\x80");
    UNIT_CHECK(ctx, app::wideToUtf8(app::utf8ToWide("plain ASCII")) == "plain ASCII");
}

void testFilterSpecs(Context & ctx)
{
    const std::vector<FileFilter> scene = app::sceneFileFilters();
    const std::vector<app::FilterSpec> specs = app::buildFilterSpecs(scene);
    UNIT_CHECK(ctx, specs.size() == scene.size());
    UNIT_CHECK(ctx, specs.size() == 2);
    UNIT_CHECK(ctx, specs[0].name == L"Gaussian splat scenes");
    // A cloud-v1 folder is opened by its manifest.json.
    UNIT_CHECK(ctx, specs[0].spec == L"*.ply;*.splat;*.spz;*.ksplat;*.sog;*.gltf;*.glb;meta.json;manifest.json");
    UNIT_CHECK(ctx, specs[1].name == L"All files");
    UNIT_CHECK(ctx, specs[1].spec == L"*.*");

    UNIT_CHECK(ctx, app::buildFilterSpecs(app::projectFileFilters())[0].spec == L"*.sabercat");
    UNIT_CHECK(ctx, app::buildFilterSpecs(app::plyFileFilters())[0].spec == L"*.ply");
    UNIT_CHECK(ctx, app::buildFilterSpecs(app::glbFileFilters())[0].spec == L"*.glb");

    // No filters, and a filter with no patterns (which means every file).
    UNIT_CHECK(ctx, app::buildFilterSpecs({}).empty());
    const std::vector<app::FilterSpec> bare = app::buildFilterSpecs({FileFilter{"Anything", ""}});
    UNIT_CHECK(ctx, bare.size() == 1);
    UNIT_CHECK(ctx, bare[0].spec == L"*.*");
    // An empty name stays empty; a non-ASCII name converts.
    const std::vector<app::FilterSpec> named = app::buildFilterSpecs(
        {FileFilter{"", "*.x"}, FileFilter{kUnicodeName, "*.y"}});
    UNIT_CHECK(ctx, named.size() == 2);
    UNIT_CHECK(ctx, named[0].name.empty());
    UNIT_CHECK(ctx, named[1].name == L"\u00DCn\u00EFc\u00F6d\u00E9 \u5C64/\u043F\u0440\u043E\u0435\u043A\u0442");
}

// A gate the runner waits on. The test opens it when it wants the dialog to
// "close".
class Latch
{
public:
    void wait()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return open_; });
    }

    void open()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            open_ = true;
        }
        cv_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    bool open_ = false;
};

// Polls until the result arrives. A dialog that never finishes ends the wait
// after ten seconds, so a broken state machine fails the check rather than
// hanging the test.
std::optional<std::optional<std::string>> pollUntilDelivered(app::AsyncFileDialog & dialog)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (std::chrono::steady_clock::now() < deadline)
    {
        std::optional<std::optional<std::string>> delivered = dialog.poll();
        if (delivered)
        {
            return delivered;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return std::nullopt;
}

const std::vector<FileFilter> kFilters = {{"Scenes", "*.ply"}};
const std::string kPath = "C:\\scenes\\room.ply";

void testAsyncPathDelivered(Context & ctx)
{
    Latch gate;
    std::atomic<int> runs{0};
    app::AsyncFileDialog dialog([&](const app::DialogRequest & request) -> std::optional<std::string>
                                {
                                    ++runs;
                                    gate.wait();
                                    return request.title == "Open scene" ? std::optional<std::string>(kPath)
                                                                        : std::nullopt;
                                });

    UNIT_CHECK(ctx, !dialog.pending());
    UNIT_CHECK(ctx, !dialog.poll().has_value());

    // Started: pending, and nothing to take while the dialog is open.
    UNIT_CHECK(ctx, dialog.startOpen(nullptr, kFilters, "Open scene", "test.open"));
    UNIT_CHECK(ctx, dialog.pending());
    UNIT_CHECK(ctx, !dialog.poll().has_value());

    // One dialog at a time: both kinds are refused while the first is pending.
    UNIT_CHECK(ctx, !dialog.startOpen(nullptr, kFilters, "Again", "test.open"));
    UNIT_CHECK(ctx, !dialog.startSave(nullptr, kFilters, "a.ply", "ply", "test.save"));
    UNIT_CHECK(ctx, !dialog.poll().has_value());

    gate.open();
    const std::optional<std::optional<std::string>> delivered = pollUntilDelivered(dialog);
    UNIT_CHECK(ctx, delivered.has_value());
    UNIT_CHECK(ctx, delivered && delivered->has_value());
    UNIT_CHECK(ctx, delivered && delivered->has_value() && **delivered == kPath);
    UNIT_CHECK(ctx, runs.load() == 1);

    // Delivered once: the result is gone and the state is idle again.
    UNIT_CHECK(ctx, !dialog.pending());
    UNIT_CHECK(ctx, !dialog.poll().has_value());
    UNIT_CHECK(ctx, !dialog.poll().has_value());
    UNIT_CHECK(ctx, runs.load() == 1);
}

void testAsyncCancelDelivered(Context & ctx)
{
    // A cancel is delivered as "done with no path": the outer optional is
    // engaged and the inner one is not.
    app::AsyncFileDialog dialog([](const app::DialogRequest &) -> std::optional<std::string>
                                { return std::nullopt; });
    UNIT_CHECK(ctx, dialog.startSave(nullptr, kFilters, "untitled.ply", "ply", "test.save"));
    const std::optional<std::optional<std::string>> delivered = pollUntilDelivered(dialog);
    UNIT_CHECK(ctx, delivered.has_value());
    UNIT_CHECK(ctx, delivered && !delivered->has_value());
    UNIT_CHECK(ctx, !dialog.pending());
    UNIT_CHECK(ctx, !dialog.poll().has_value());
}

void testAsyncFailureDelivered(Context & ctx)
{
    // A runner that throws is a dialog that could not be shown: the app gets a
    // cancel, never an exception on the UI thread.
    app::AsyncFileDialog dialog([](const app::DialogRequest &) -> std::optional<std::string>
                                { throw std::runtime_error("no dialog here"); });
    UNIT_CHECK(ctx, dialog.startOpen(nullptr, kFilters, "Open", "test.open"));
    const std::optional<std::optional<std::string>> delivered = pollUntilDelivered(dialog);
    UNIT_CHECK(ctx, delivered && !delivered->has_value());
    UNIT_CHECK(ctx, !dialog.pending());
}

void testAsyncDestroyWhilePending(Context & ctx)
{
    // Destroying the object with a dialog still running joins its worker: the
    // destructor returns only after the runner has finished.
    Latch gate;
    std::atomic<bool> finished{false};
    std::thread releaser;
    {
        app::AsyncFileDialog dialog([&](const app::DialogRequest &) -> std::optional<std::string>
                                    {
                                        gate.wait();
                                        finished = true;
                                        return kPath;
                                    });
        UNIT_CHECK(ctx, dialog.startOpen(nullptr, kFilters, "Open", "test.open"));
        releaser = std::thread([&gate] {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            gate.open();
        });
        // Leaves scope while pending and never polled.
    }
    UNIT_CHECK(ctx, finished.load());
    releaser.join();
}

} // namespace

void testFileDialog(Context & ctx)
{
    testDefaultExtension(ctx);
#ifdef _WIN32
    testUtf8Wide(ctx);
    testFilterSpecs(ctx);
#else
    // The wide-string conversions are Windows-only (the apps are too).
    (void)testUtf8Wide;
    (void)testFilterSpecs;
#endif
    testAsyncPathDelivered(ctx);
    testAsyncCancelDelivered(ctx);
    testAsyncFailureDelivered(ctx);
    testAsyncDestroyWhilePending(ctx);
}

} // namespace sabercat::unit
