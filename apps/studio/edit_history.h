// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Sabercat Studio's undo/redo: a linear history of reversible edits with a byte
// budget. Plain C++ with no GPU or window code, so the CPU unit tests link it.
// A command owns whatever it needs to reverse itself (the Studio's commands keep
// snapshots of the data they touch). Not thread-safe: the Studio calls it from
// the UI thread only.
#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>

namespace sabercat::studio
{

// One reversible edit. The edit has already been executed when the command is
// pushed, so push() never calls undo() or redo(). redo() must re-apply the same
// edit to the state that undo() restores.
class EditCommand
{
  public:
    virtual ~EditCommand() = default;

    // Reverses the edit.
    virtual void undo() = 0;
    // Re-applies the edit.
    virtual void redo() = 0;
    // A short name for the edit, e.g. "Delete 3 layers". Queried once, at push.
    virtual std::string label() const = 0;
    // Host memory the command holds, counted against the history's budget.
    // Queried once, at push.
    virtual std::size_t bytes() const = 0;
};

// The commands, oldest first, with a cursor between them: the commands before
// the cursor are applied (undo() reverses the one just before it), the ones at
// and after it are the redo tail. Index 0 is before the first command and
// size() is after the last.
class EditHistory
{
  public:
    // budgetBytes caps the total bytes() of the held commands (1 GiB by default).
    explicit EditHistory(std::size_t budgetBytes = std::size_t{1} << 30);
    EditHistory(const EditHistory &) = delete;
    EditHistory & operator=(const EditHistory &) = delete;

    // Records a command that has already been executed. A null command throws
    // std::invalid_argument and changes nothing. Otherwise: the redo tail (the
    // commands at index >= cursor()) is destroyed without any undo or redo call,
    // the command is appended, cursor() becomes size(), and the budget is
    // enforced (see setBudget). If label() or bytes() throws, nothing changes.
    void push(std::unique_ptr<EditCommand> command);

    // canUndo: cursor() > 0. canRedo: cursor() < size().
    bool canUndo() const;
    bool canRedo() const;

    // Undoes the command before the cursor: calls its undo(), then moves the
    // cursor back one. Returns false, changing nothing, when !canUndo(). If
    // undo() throws, the cursor is unchanged and the exception propagates.
    bool undo();
    // Mirror of undo(): calls redo() on the command at the cursor, then moves the
    // cursor forward one. Returns false, changing nothing, when !canRedo().
    bool redo();

    // Moves the cursor to `index` (0 to size()) with repeated undo() or redo()
    // calls. Returns false, changing nothing, when index > size(). If a step
    // throws, the exception propagates with the cursor at the last completed step.
    bool jumpTo(std::size_t index);

    // Destroys every command without calling undo() or redo() (New or Open).
    // The cursor and bytesUsed() become 0. evictedCount() is unchanged: dropping
    // everything is not an eviction.
    void clear();

    std::size_t size() const;
    std::size_t cursor() const;
    // The sum of bytes() over the held commands.
    std::size_t bytesUsed() const;
    std::size_t budget() const;

    // Sets the budget, then evicts as push() does when bytesUsed() is over it.
    // Eviction takes the oldest command while more than one remains. Each
    // eviction logs one warning containing "edit history" and the label, counts
    // toward evictedCount(), and moves the cursor down by one. If the cursor is 0
    // (everything undone) and the history is over budget, the redo tail is
    // destroyed instead, silently and without counting as an eviction: the oldest
    // command cannot go while its redo chain still depends on it.
    // A single command left over budget is kept, with one warning containing
    // "exceeds the budget" per call.
    void setBudget(std::size_t bytes);

    // The label captured at push. Throws std::out_of_range for index >= size().
    const std::string & label(std::size_t index) const;
    // The bytes() captured at push. Throws std::out_of_range for index >= size().
    std::size_t entryBytes(std::size_t index) const;
    // Commands the budget has evicted since construction. clear() does not reset it.
    std::uint64_t evictedCount() const;

  private:
    // Plain data, no padding: the pointer and the string are 8-byte aligned.
    struct Entry
    {
        std::unique_ptr<EditCommand> command;
        std::string label;
        std::size_t bytes;
    };

    const Entry & entry(std::size_t index) const;
    void evictOldest();
    void enforceBudget();

    std::deque<Entry> m_entries; // a deque: eviction from the front is O(1)
    std::size_t m_cursor = 0;
    std::size_t m_bytesUsed = 0;
    std::size_t m_budget;
    std::uint64_t m_evictedCount = 0;
};

} // namespace sabercat::studio
