// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// The Studio's undo/redo history (apps/studio/edit_history.h), GPU-free: the
// order of calls, the redo tail, jumps, the byte budget and its warnings, clear,
// exception safety, and a bit-exact restore over buffer commands that mimic the
// Studio's real ones (transform, delete, duplicate).
#include "unit_test.h"

#include "edit_history.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <utility>
#include <vector>

namespace sabercat::unit
{

namespace
{

using studio::EditCommand;
using studio::EditHistory;

// Holds std::cerr (where Logger::warning writes) so a test can count the lines
// the code logged. Restores the stream when it goes out of scope.
class CerrCapture
{
  public:
    CerrCapture() : m_saved(std::cerr.rdbuf(m_text.rdbuf())) {}
    ~CerrCapture()
    {
        std::cerr.rdbuf(m_saved);
    }
    CerrCapture(const CerrCapture &) = delete;
    CerrCapture & operator=(const CerrCapture &) = delete;

    // Lines that contain `needle`.
    int count(const std::string & needle) const
    {
        std::istringstream lines(m_text.str());
        std::string line;
        int matches = 0;
        while (std::getline(lines, line))
        {
            matches += line.find(needle) != std::string::npos ? 1 : 0;
        }
        return matches;
    }

  private:
    std::ostringstream m_text;
    std::streambuf * m_saved;
};

// --- Counting commands ----------------------------------------------------

// What the fake commands report back to the test. Outlives the history.
struct Record
{
    std::vector<std::string> calls;     // "undo:A" and "redo:B", in call order
    std::vector<std::string> destroyed; // labels, in destruction order
    int labelCalls = 0;                 // label() queries
    int bytesCalls = 0;                 // bytes() queries
};

// Logs its undo and redo calls; can be told to throw from either.
class CountingCommand final : public EditCommand
{
  public:
    CountingCommand(std::string label, std::size_t bytes, Record & record)
        : m_label(std::move(label)), m_bytes(bytes), m_record(record)
    {
    }
    ~CountingCommand() override
    {
        m_record.destroyed.push_back(m_label);
    }

    void setFailures(bool undoThrows, bool redoThrows)
    {
        m_undoThrows = undoThrows;
        m_redoThrows = redoThrows;
    }

    void undo() override
    {
        if (m_undoThrows)
        {
            throw std::runtime_error("undo failed");
        }
        m_record.calls.push_back("undo:" + m_label);
    }
    void redo() override
    {
        if (m_redoThrows)
        {
            throw std::runtime_error("redo failed");
        }
        m_record.calls.push_back("redo:" + m_label);
    }
    std::string label() const override
    {
        ++m_record.labelCalls;
        return m_label;
    }
    std::size_t bytes() const override
    {
        ++m_record.bytesCalls;
        return m_bytes;
    }

  private:
    std::string m_label;
    std::size_t m_bytes;
    Record & m_record;
    bool m_undoThrows = false;
    bool m_redoThrows = false;
};

std::unique_ptr<EditCommand> makeCounting(const char * label, std::size_t bytes, Record & record)
{
    return std::make_unique<CountingCommand>(label, bytes, record);
}

// --- Basics and ordering --------------------------------------------------

void testBasics(Context & ctx)
{
    Record record;
    EditHistory history;
    UNIT_CHECK(ctx, !history.canUndo() && !history.canRedo());
    UNIT_CHECK(ctx, !history.undo() && !history.redo());
    UNIT_CHECK(ctx, history.jumpTo(0));
    UNIT_CHECK(ctx, !history.jumpTo(1));
    UNIT_CHECK(ctx, history.size() == 0 && history.cursor() == 0 && history.bytesUsed() == 0);

    bool threw = false;
    try
    {
        history.push(nullptr);
    }
    catch (const std::invalid_argument &)
    {
        threw = true;
    }
    UNIT_CHECK(ctx, threw && history.size() == 0);

    history.push(makeCounting("A", 10, record));
    history.push(makeCounting("B", 20, record));
    history.push(makeCounting("C", 30, record));
    UNIT_CHECK(ctx, history.size() == 3 && history.cursor() == 3);
    UNIT_CHECK(ctx, history.canUndo() && !history.canRedo());
    UNIT_CHECK(ctx, record.calls.empty()); // push never calls undo or redo
    UNIT_CHECK(ctx, history.label(0) == "A" && history.label(1) == "B" && history.label(2) == "C");
    UNIT_CHECK(ctx, history.entryBytes(0) == 10 && history.entryBytes(1) == 20 && history.entryBytes(2) == 30);
    UNIT_CHECK(ctx, history.bytesUsed() == 60 && history.evictedCount() == 0);

    // Labels and sizes are queried once per push, and the accessors use the cache.
    UNIT_CHECK(ctx, record.labelCalls == 3 && record.bytesCalls == 3);
    (void)history.label(0);
    (void)history.entryBytes(2);
    UNIT_CHECK(ctx, record.labelCalls == 3 && record.bytesCalls == 3);

    threw = false;
    try
    {
        (void)history.label(3);
    }
    catch (const std::out_of_range &)
    {
        threw = true;
    }
    UNIT_CHECK(ctx, threw);
    threw = false;
    try
    {
        (void)history.entryBytes(3);
    }
    catch (const std::out_of_range &)
    {
        threw = true;
    }
    UNIT_CHECK(ctx, threw);

    UNIT_CHECK(ctx, history.undo());
    UNIT_CHECK(ctx, history.undo());
    UNIT_CHECK(ctx, history.cursor() == 1 && history.canUndo() && history.canRedo());
    UNIT_CHECK(ctx, (record.calls == std::vector<std::string>{"undo:C", "undo:B"}));
    UNIT_CHECK(ctx, history.redo());
    UNIT_CHECK(ctx, history.cursor() == 2);
    UNIT_CHECK(ctx, record.calls.back() == "redo:B");
    UNIT_CHECK(ctx, record.calls.size() == 3);
}

// Pushing after an undo drops the redo tail: its commands are destroyed, not undone.
void testRedoTailDrop(Context & ctx)
{
    Record record;
    EditHistory history;
    history.push(makeCounting("A", 100, record));
    history.push(makeCounting("B", 200, record));
    history.push(makeCounting("C", 300, record));
    UNIT_CHECK(ctx, history.undo());
    UNIT_CHECK(ctx, history.undo());
    UNIT_CHECK(ctx, record.destroyed.empty());

    history.push(makeCounting("D", 400, record));
    UNIT_CHECK(ctx, history.size() == 2 && history.cursor() == 2);
    UNIT_CHECK(ctx, !history.canRedo());
    UNIT_CHECK(ctx, history.label(0) == "A" && history.label(1) == "D");
    UNIT_CHECK(ctx, history.bytesUsed() == 500);
    UNIT_CHECK(ctx, (record.destroyed == std::vector<std::string>{"C", "B"}));
    UNIT_CHECK(ctx, (record.calls == std::vector<std::string>{"undo:C", "undo:B"})); // no new calls
}

void testJumpTo(Context & ctx)
{
    Record record;
    EditHistory history;
    for (const char * label : {"A", "B", "C", "D", "E"})
    {
        history.push(makeCounting(label, 1, record));
    }
    UNIT_CHECK(ctx, history.cursor() == 5);
    record.calls.clear();

    UNIT_CHECK(ctx, history.jumpTo(2));
    UNIT_CHECK(ctx, history.cursor() == 2);
    UNIT_CHECK(ctx, (record.calls == std::vector<std::string>{"undo:E", "undo:D", "undo:C"}));

    record.calls.clear();
    UNIT_CHECK(ctx, history.jumpTo(4));
    UNIT_CHECK(ctx, history.cursor() == 4);
    UNIT_CHECK(ctx, (record.calls == std::vector<std::string>{"redo:C", "redo:D"}));

    record.calls.clear();
    UNIT_CHECK(ctx, history.jumpTo(0));
    UNIT_CHECK(ctx, history.cursor() == 0);
    UNIT_CHECK(ctx, (record.calls == std::vector<std::string>{"undo:D", "undo:C", "undo:B", "undo:A"}));

    record.calls.clear();
    UNIT_CHECK(ctx, history.jumpTo(5));
    UNIT_CHECK(ctx, history.cursor() == 5);
    UNIT_CHECK(ctx, (record.calls == std::vector<std::string>{"redo:A", "redo:B", "redo:C", "redo:D", "redo:E"}));

    // Out of range: false, no calls, no change. In range but already there: true, no calls.
    record.calls.clear();
    UNIT_CHECK(ctx, !history.jumpTo(6));
    UNIT_CHECK(ctx, history.cursor() == 5 && record.calls.empty());
    UNIT_CHECK(ctx, history.jumpTo(5) && record.calls.empty());
}

// --- Budget ---------------------------------------------------------------

void testBudgetEviction(Context & ctx)
{
    Record record;
    EditHistory history(1000);
    {
        CerrCapture capture;
        history.push(makeCounting("A", 400, record));
        history.push(makeCounting("B", 400, record));
        UNIT_CHECK(ctx, capture.count("edit history") == 0);
        UNIT_CHECK(ctx, history.size() == 2 && history.bytesUsed() == 800);

        history.push(makeCounting("C", 400, record));
        UNIT_CHECK(ctx, capture.count("edit history") == 1);
        UNIT_CHECK(ctx, capture.count("\"A\"") == 1);
        UNIT_CHECK(ctx, capture.count("exceeds the budget") == 0);
    }
    UNIT_CHECK(ctx, history.size() == 2 && history.cursor() == 2 && history.bytesUsed() == 800);
    UNIT_CHECK(ctx, history.evictedCount() == 1);
    UNIT_CHECK(ctx, history.label(0) == "B" && history.label(1) == "C");

    {
        CerrCapture capture;
        history.push(makeCounting("D", 900, record));
        // B and C go, oldest first, to reach 900 <= 1000.
        UNIT_CHECK(ctx, capture.count("edit history") == 2);
        UNIT_CHECK(ctx, capture.count("\"B\"") == 1 && capture.count("\"C\"") == 1);
        UNIT_CHECK(ctx, capture.count("exceeds the budget") == 0);
    }
    UNIT_CHECK(ctx, history.size() == 1 && history.cursor() == 1 && history.label(0) == "D");
    UNIT_CHECK(ctx, history.bytesUsed() == 900 && history.evictedCount() == 3);

    // One entry bigger than the whole budget is kept, with an "exceeds" warning.
    {
        CerrCapture capture;
        history.push(makeCounting("E", 5000, record));
        UNIT_CHECK(ctx, capture.count("edit history") == 1 && capture.count("\"D\"") == 1);
        UNIT_CHECK(ctx, capture.count("exceeds the budget") == 1);
    }
    UNIT_CHECK(ctx, history.size() == 1 && history.label(0) == "E" && history.bytesUsed() == 5000);
    UNIT_CHECK(ctx, history.evictedCount() == 4);

    // Pushing a small entry onto it evicts the oversized one; no warning about the budget.
    {
        CerrCapture capture;
        history.push(makeCounting("F", 100, record));
        UNIT_CHECK(ctx, capture.count("edit history") == 1 && capture.count("\"E\"") == 1);
        UNIT_CHECK(ctx, capture.count("exceeds the budget") == 0);
    }
    UNIT_CHECK(ctx, history.size() == 1 && history.label(0) == "F" && history.bytesUsed() == 100);
    UNIT_CHECK(ctx, history.evictedCount() == 5);

    // An oversized first push is kept, with one warning.
    EditHistory fresh(1000);
    {
        CerrCapture capture;
        fresh.push(makeCounting("G", 5000, record));
        UNIT_CHECK(ctx, capture.count("exceeds the budget") == 1);
        UNIT_CHECK(ctx, capture.count("edit history") == 0);
    }
    UNIT_CHECK(ctx, fresh.size() == 1 && fresh.evictedCount() == 0);
}

// setBudget lowering evicts oldest first, with the same warnings as push.
void testSetBudget(Context & ctx)
{
    Record record;
    EditHistory history(1000);
    history.push(makeCounting("X", 300, record));
    history.push(makeCounting("Y", 300, record));
    history.push(makeCounting("Z", 300, record));
    UNIT_CHECK(ctx, history.evictedCount() == 0);

    {
        CerrCapture capture;
        history.setBudget(500);
        UNIT_CHECK(ctx, capture.count("edit history") == 2);
        UNIT_CHECK(ctx, capture.count("\"X\"") == 1 && capture.count("\"Y\"") == 1);
        UNIT_CHECK(ctx, capture.count("exceeds the budget") == 0);
    }
    UNIT_CHECK(ctx, history.budget() == 500);
    UNIT_CHECK(ctx, history.size() == 1 && history.cursor() == 1 && history.label(0) == "Z");
    UNIT_CHECK(ctx, history.bytesUsed() == 300 && history.evictedCount() == 2);

    // Raising the budget evicts nothing.
    {
        CerrCapture capture;
        history.setBudget(10000);
        UNIT_CHECK(ctx, capture.count("edit history") == 0);
    }
    UNIT_CHECK(ctx, history.size() == 1 && history.evictedCount() == 2);

    // With everything undone, an over-budget history drops the redo tail instead of
    // evicting (the oldest entry cannot go while the redo chain depends on it).
    EditHistory undone(1000);
    Record undoneRecord;
    undone.push(makeCounting("P", 300, undoneRecord));
    undone.push(makeCounting("Q", 300, undoneRecord));
    UNIT_CHECK(ctx, undone.undo() && undone.undo());
    UNIT_CHECK(ctx, undone.cursor() == 0);
    {
        CerrCapture capture;
        undone.setBudget(100);
        UNIT_CHECK(ctx, capture.count("edit history") == 0);
    }
    UNIT_CHECK(ctx, undone.size() == 0 && undone.cursor() == 0 && undone.bytesUsed() == 0);
    UNIT_CHECK(ctx, undone.evictedCount() == 0);
    UNIT_CHECK(ctx, undoneRecord.destroyed.size() == 2);
}

// Push drops the redo tail before it evicts, so the cursor always ends at size().
void testEvictionWithCursorBelowEnd(Context & ctx)
{
    Record record;
    EditHistory history(1000);
    history.push(makeCounting("A", 300, record));
    history.push(makeCounting("B", 300, record));
    history.push(makeCounting("C", 300, record));
    UNIT_CHECK(ctx, history.undo() && history.undo());
    UNIT_CHECK(ctx, history.cursor() == 1 && history.size() == 3);

    {
        CerrCapture capture;
        history.push(makeCounting("Big", 800, record));
        // B and C are destroyed as the redo tail; then A (1100 > 1000) is evicted and destroyed too.
        UNIT_CHECK(ctx, capture.count("edit history") == 1 && capture.count("\"A\"") == 1);
    }
    UNIT_CHECK(ctx, history.size() == 1 && history.cursor() == 1 && history.cursor() == history.size());
    UNIT_CHECK(ctx, history.label(0) == "Big" && history.bytesUsed() == 800);
    UNIT_CHECK(ctx, history.evictedCount() == 1);
    UNIT_CHECK(ctx, (record.destroyed == std::vector<std::string>{"C", "B", "A"}));
    UNIT_CHECK(ctx, !history.canRedo());
}

// --- clear, exceptions ----------------------------------------------------

void testClear(Context & ctx)
{
    Record record;
    EditHistory history(1000);
    {
        CerrCapture capture;
        for (const char * label : {"A", "B", "C", "D"})
        {
            history.push(makeCounting(label, 400, record));
        }
    }
    // A and B were evicted by the budget (1200 > 1000 twice).
    UNIT_CHECK(ctx, history.evictedCount() == 2 && history.size() == 2);
    UNIT_CHECK(ctx, history.undo());
    record.calls.clear();
    record.destroyed.clear();

    history.clear();
    UNIT_CHECK(ctx, history.size() == 0 && history.cursor() == 0 && history.bytesUsed() == 0);
    UNIT_CHECK(ctx, !history.canUndo() && !history.canRedo());
    UNIT_CHECK(ctx, record.calls.empty()); // dropping is not undoing
    UNIT_CHECK(ctx, record.destroyed.size() == 2); // C and D
    UNIT_CHECK(ctx, history.evictedCount() == 2);
    UNIT_CHECK(ctx, history.jumpTo(0) && !history.jumpTo(1));
}

void testExceptionSafety(Context & ctx)
{
    Record record;
    EditHistory history;

    // undo() throws: the exception propagates and the cursor stays where it was.
    history.push(makeCounting("A", 10, record));
    history.push(makeCounting("B", 10, record));
    {
        auto failing = std::make_unique<CountingCommand>("C", 10, record);
        failing->setFailures(true, false);
        history.push(std::move(failing));
    }
    bool threw = false;
    try
    {
        history.undo();
    }
    catch (const std::runtime_error &)
    {
        threw = true;
    }
    UNIT_CHECK(ctx, threw && history.cursor() == 3);
    UNIT_CHECK(ctx, record.calls.empty());

    // redo() throws: the cursor stays before the failing command.
    history.clear();
    record.calls.clear();
    history.push(makeCounting("A", 10, record));
    {
        auto failing = std::make_unique<CountingCommand>("B", 10, record);
        failing->setFailures(false, true);
        history.push(std::move(failing));
    }
    UNIT_CHECK(ctx, history.undo()); // B's undo works
    UNIT_CHECK(ctx, history.cursor() == 1 && history.canRedo());
    threw = false;
    try
    {
        history.redo();
    }
    catch (const std::runtime_error &)
    {
        threw = true;
    }
    UNIT_CHECK(ctx, threw && history.cursor() == 1 && history.canRedo());
    UNIT_CHECK(ctx, history.size() == 2);
}

// --- Bit-exact restore over buffer commands -------------------------------

// Each buffer command edits a byte vector it points to. apply() is the execute
// step, and redo() calls it again. undo() restores what the command saved.

constexpr std::size_t kOps = 200;
constexpr std::size_t kStartBytes = 10007;
constexpr std::size_t kMinBytes = 64;     // deletes that would leave fewer are skipped
constexpr std::size_t kMaxBytes = 40000;  // duplicates stop above this size

// xorshift32; the state must not be zero.
std::uint32_t nextRandom(std::uint32_t & state)
{
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

// A deterministic integer hash of (index, seed).
std::uint32_t selectionHash(std::size_t index, std::uint32_t seed)
{
    std::uint32_t h = seed ^ (static_cast<std::uint32_t>(index) * 0x9E3779B1u);
    h ^= h >> 16;
    h *= 0x7FEB352Du;
    h ^= h >> 15;
    h *= 0x846CA68Bu;
    h ^= h >> 16;
    return h;
}

// Every index i in [0, size) with hash(i, seed) % 3 == 0, ascending.
std::vector<std::size_t> selectBytes(std::size_t size, std::uint32_t seed)
{
    std::vector<std::size_t> picked;
    for (std::size_t i = 0; i < size; ++i)
    {
        if (selectionHash(i, seed) % 3 == 0)
        {
            picked.push_back(i);
        }
    }
    return picked;
}

std::uint8_t rotl8(std::uint8_t value, int count)
{
    return static_cast<std::uint8_t>((value << count) | (value >> (8 - count)));
}

// The transform kernel: depends only on the key and the input byte.
std::uint8_t transformByte(std::uint8_t value, std::uint8_t key)
{
    return static_cast<std::uint8_t>(rotl8(static_cast<std::uint8_t>(value ^ key), 3) + 1);
}

bool sameBytes(const std::vector<std::uint8_t> & a, const std::vector<std::uint8_t> & b)
{
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size()) == 0);
}

class BufferCommand : public EditCommand
{
  public:
    explicit BufferCommand(std::vector<std::uint8_t> & data) : m_data(data) {}

    // Runs the edit on the current buffer.
    virtual void apply() = 0;
    void redo() override
    {
        apply();
    }

  protected:
    std::vector<std::uint8_t> & m_data;
};

// Rewrites the picked bytes. Keeps their values from before the edit (sparse snapshot).
class TransformCommand final : public BufferCommand
{
  public:
    TransformCommand(std::vector<std::uint8_t> & data, std::vector<std::size_t> picked, std::uint8_t key)
        : BufferCommand(data), m_picked(std::move(picked)), m_key(key)
    {
        m_saved.reserve(m_picked.size());
        for (std::size_t index : m_picked)
        {
            m_saved.push_back(m_data[index]);
        }
    }

    void apply() override
    {
        for (std::size_t index : m_picked)
        {
            m_data[index] = transformByte(m_data[index], m_key);
        }
    }
    void undo() override
    {
        for (std::size_t k = 0; k < m_picked.size(); ++k)
        {
            m_data[m_picked[k]] = m_saved[k];
        }
    }
    std::string label() const override
    {
        return "transform";
    }
    std::size_t bytes() const override
    {
        return sizeof(*this) + m_picked.size() * sizeof(std::size_t) + m_saved.size();
    }

  private:
    std::vector<std::size_t> m_picked;
    std::vector<std::uint8_t> m_saved;
    std::uint8_t m_key;
};

// Erases the picked bytes, keeping them and a keep mask over the old buffer.
class DeleteCommand final : public BufferCommand
{
  public:
    DeleteCommand(std::vector<std::uint8_t> & data, const std::vector<std::size_t> & picked)
        : BufferCommand(data), m_keep(data.size(), true)
    {
        for (std::size_t index : picked)
        {
            m_keep[index] = false;
            m_removed.push_back(m_data[index]);
        }
    }

    void apply() override
    {
        std::size_t out = 0;
        for (std::size_t p = 0; p < m_keep.size(); ++p)
        {
            if (m_keep[p])
            {
                m_data[out++] = m_data[p];
            }
        }
        m_data.resize(out);
    }
    // Rebuilds the original order from the kept bytes and the saved ones.
    void undo() override
    {
        std::vector<std::uint8_t> full(m_keep.size());
        std::size_t kept = 0;
        std::size_t gone = 0;
        for (std::size_t p = 0; p < full.size(); ++p)
        {
            full[p] = m_keep[p] ? m_data[kept++] : m_removed[gone++];
        }
        m_data = std::move(full);
    }
    std::string label() const override
    {
        return "delete";
    }
    std::size_t bytes() const override
    {
        return sizeof(*this) + m_keep.size() / 8 + m_removed.size();
    }

  private:
    std::vector<bool> m_keep;
    std::vector<std::uint8_t> m_removed;
};

// Appends copies of the picked bytes to the end.
class DuplicateCommand final : public BufferCommand
{
  public:
    DuplicateCommand(std::vector<std::uint8_t> & data, std::vector<std::size_t> picked)
        : BufferCommand(data), m_oldSize(data.size()), m_picked(std::move(picked))
    {
    }

    void apply() override
    {
        for (std::size_t index : m_picked)
        {
            const std::uint8_t value = m_data[index];
            m_data.push_back(value);
        }
    }
    void undo() override
    {
        m_data.resize(m_oldSize);
    }
    std::string label() const override
    {
        return "duplicate";
    }
    std::size_t bytes() const override
    {
        return sizeof(*this) + m_picked.size() * sizeof(std::size_t);
    }

  private:
    std::size_t m_oldSize;
    std::vector<std::size_t> m_picked;
};

void testBitExactRestore(Context & ctx)
{
    std::vector<std::uint8_t> data(kStartBytes);
    std::uint32_t rng = 0x2545F491u;
    for (std::uint8_t & byte : data)
    {
        byte = static_cast<std::uint8_t>(nextRandom(rng) >> 24);
    }

    // states[i] is the buffer after i operations.
    std::vector<std::vector<std::uint8_t>> states;
    states.reserve(kOps + 1);
    states.push_back(data);

    EditHistory history;
    std::size_t transforms = 0;
    std::size_t deletes = 0;
    std::size_t duplicates = 0;
    for (std::size_t op = 0; op < kOps; ++op)
    {
        const std::uint32_t seed = nextRandom(rng);
        const auto key = static_cast<std::uint8_t>(nextRandom(rng));
        std::uint32_t kind = nextRandom(rng) % 3;
        std::vector<std::size_t> picked = selectBytes(data.size(), seed);
        if (kind == 1 && data.size() - picked.size() < kMinBytes)
        {
            kind = 0; // skip this delete: a transform instead
        }
        if (kind == 2 && data.size() > kMaxBytes)
        {
            kind = 0;
        }

        // Execute, then push.
        if (kind == 0)
        {
            auto command = std::make_unique<TransformCommand>(data, std::move(picked), key);
            command->apply();
            history.push(std::move(command));
            ++transforms;
        }
        else if (kind == 1)
        {
            auto command = std::make_unique<DeleteCommand>(data, picked);
            command->apply();
            history.push(std::move(command));
            ++deletes;
        }
        else
        {
            auto command = std::make_unique<DuplicateCommand>(data, std::move(picked));
            command->apply();
            history.push(std::move(command));
            ++duplicates;
        }
        states.push_back(data);
    }
    UNIT_CHECK(ctx, transforms > 0 && deletes > 0 && duplicates > 0); // all three kinds ran
    UNIT_CHECK(ctx, history.size() == kOps && history.cursor() == kOps);
    UNIT_CHECK(ctx, data.size() >= kMinBytes);

    // (i) Undo one at a time; after each undo the buffer is the earlier state.
    for (std::size_t k = 0; k < kOps; ++k)
    {
        UNIT_CHECK(ctx, history.undo());
        UNIT_CHECK(ctx, sameBytes(data, states[kOps - 1 - k]));
    }
    // (ii) All undone: back at the start.
    UNIT_CHECK(ctx, !history.canUndo() && history.cursor() == 0);
    UNIT_CHECK(ctx, sameBytes(data, states[0]));

    // (iii) Redo one at a time.
    for (std::size_t k = 0; k < kOps; ++k)
    {
        UNIT_CHECK(ctx, history.redo());
        UNIT_CHECK(ctx, sameBytes(data, states[k + 1]));
    }
    // (iv) All redone: the final state.
    UNIT_CHECK(ctx, !history.canRedo() && history.cursor() == kOps);
    UNIT_CHECK(ctx, sameBytes(data, states[kOps]));

    // (v) Random jumps land on the matching state.
    for (std::size_t n = 0; n < 300; ++n)
    {
        const std::size_t index = nextRandom(rng) % (kOps + 1);
        UNIT_CHECK(ctx, history.jumpTo(index));
        UNIT_CHECK(ctx, history.cursor() == index);
        UNIT_CHECK(ctx, sameBytes(data, states[index]));
    }
}

} // namespace

void testEditHistory(Context & ctx)
{
    testBasics(ctx);
    testRedoTailDrop(ctx);
    testJumpTo(ctx);
    testBudgetEviction(ctx);
    testSetBudget(ctx);
    testEvictionWithCursorBelowEnd(ctx);
    testClear(ctx);
    testExceptionSafety(ctx);
    testBitExactRestore(ctx);
}

} // namespace sabercat::unit
