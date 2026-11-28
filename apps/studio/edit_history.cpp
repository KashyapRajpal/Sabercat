// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "edit_history.h"

#include "sabercat/logger.h"

#include <stdexcept>
#include <utility>

namespace sabercat::studio
{

EditHistory::EditHistory(std::size_t budgetBytes) : m_budget(budgetBytes) {}

void EditHistory::push(std::unique_ptr<EditCommand> command)
{
    if (!command)
    {
        throw std::invalid_argument("EditHistory::push: null command");
    }
    // Query before touching the history, so a throwing query leaves it unchanged.
    std::string label = command->label();
    const std::size_t bytes = command->bytes();

    // Drop the redo tail; its destructors run here (no undo or redo calls).
    while (m_entries.size() > m_cursor)
    {
        m_bytesUsed -= m_entries.back().bytes;
        m_entries.pop_back();
    }

    m_entries.push_back(Entry{std::move(command), std::move(label), bytes});
    m_bytesUsed += bytes;
    m_cursor = m_entries.size();
    enforceBudget();
}

bool EditHistory::canUndo() const
{
    return m_cursor > 0;
}

bool EditHistory::canRedo() const
{
    return m_cursor < m_entries.size();
}

bool EditHistory::undo()
{
    if (!canUndo())
    {
        return false;
    }
    m_entries[m_cursor - 1].command->undo();
    --m_cursor;
    return true;
}

bool EditHistory::redo()
{
    if (!canRedo())
    {
        return false;
    }
    m_entries[m_cursor].command->redo();
    ++m_cursor;
    return true;
}

bool EditHistory::jumpTo(std::size_t index)
{
    if (index > m_entries.size())
    {
        return false;
    }
    while (m_cursor > index)
    {
        undo();
    }
    while (m_cursor < index)
    {
        redo();
    }
    return true;
}

void EditHistory::clear()
{
    m_entries.clear();
    m_cursor = 0;
    m_bytesUsed = 0;
}

std::size_t EditHistory::size() const
{
    return m_entries.size();
}

std::size_t EditHistory::cursor() const
{
    return m_cursor;
}

std::size_t EditHistory::bytesUsed() const
{
    return m_bytesUsed;
}

std::size_t EditHistory::budget() const
{
    return m_budget;
}

void EditHistory::setBudget(std::size_t bytes)
{
    m_budget = bytes;
    enforceBudget();
}

const std::string & EditHistory::label(std::size_t index) const
{
    return entry(index).label;
}

std::size_t EditHistory::entryBytes(std::size_t index) const
{
    return entry(index).bytes;
}

std::uint64_t EditHistory::evictedCount() const
{
    return m_evictedCount;
}

const EditHistory::Entry & EditHistory::entry(std::size_t index) const
{
    if (index >= m_entries.size())
    {
        throw std::out_of_range("EditHistory: index " + std::to_string(index) + " is out of range");
    }
    return m_entries[index];
}

// Removes the oldest command, which must be applied (cursor() > 0).
void EditHistory::evictOldest()
{
    const Entry & oldest = m_entries.front();
    Logger::warning("edit history: evicted \"" + oldest.label + "\" (" + std::to_string(oldest.bytes) +
                    " bytes) to stay within the budget of " + std::to_string(m_budget) + " bytes");
    m_bytesUsed -= oldest.bytes;
    m_entries.pop_front();
    --m_cursor;
    ++m_evictedCount;
}

void EditHistory::enforceBudget()
{
    while (m_bytesUsed > m_budget && m_entries.size() > 1)
    {
        if (m_cursor == 0)
        {
            // Nothing is applied, so the oldest entry cannot go without breaking
            // the redo chain: drop the redo tail instead (see setBudget).
            m_entries.clear();
            m_bytesUsed = 0;
            break;
        }
        evictOldest();
    }
    if (m_entries.size() == 1 && m_bytesUsed > m_budget)
    {
        Logger::warning("Studio: undo entry \"" + m_entries.front().label + "\" (" +
                        std::to_string(m_entries.front().bytes) + " bytes) exceeds the budget of " +
                        std::to_string(m_budget) + " bytes and is kept");
    }
}

} // namespace sabercat::studio
