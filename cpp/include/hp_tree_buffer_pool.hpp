#pragma once

#include "hp_tree_common.hpp"
#include <list>
#include <fstream>

namespace hptree {

struct DiskPage {
    PageId   page_id    = INVALID_PAGE;
    bool     dirty      = false;
    uint32_t pin_count  = 0;
    LSN      page_lsn   = INVALID_LSN;
    std::vector<uint8_t> data;

    DiskPage() = default;
    explicit DiskPage(size_t page_size) : data(page_size, 0) {}
};

struct PageHeader {
    PageId   page_id     = INVALID_PAGE;
    NodeType node_type   = NodeType::LEAF;
    uint32_t record_count= 0;
    uint32_t free_space  = 0;
    PageId   next_page   = INVALID_PAGE;
    PageId   prev_page   = INVALID_PAGE;
    LSN      page_lsn    = INVALID_LSN;
    uint8_t  reserved[16]= {};

    static constexpr size_t SIZE = sizeof(PageId) * 3 + sizeof(NodeType)
        + sizeof(uint32_t) * 2 + sizeof(LSN) + 16;

    void serialize_into(uint8_t* buf) const {
        size_t off = 0;
        auto w = [&](const void* p, size_t n) {
            std::memcpy(buf + off, p, n); off += n;
        };
        w(&page_id, sizeof(page_id));
        w(&node_type, sizeof(node_type));
        w(&record_count, sizeof(record_count));
        w(&free_space, sizeof(free_space));
        w(&next_page, sizeof(next_page));
        w(&prev_page, sizeof(prev_page));
        w(&page_lsn, sizeof(page_lsn));
        w(reserved, sizeof(reserved));
    }

    static PageHeader deserialize_from(const uint8_t* buf) {
        PageHeader h;
        size_t off = 0;
        auto r = [&](void* p, size_t n) {
            std::memcpy(p, buf + off, n); off += n;
        };
        r(&h.page_id, sizeof(h.page_id));
        r(&h.node_type, sizeof(h.node_type));
        r(&h.record_count, sizeof(h.record_count));
        r(&h.free_space, sizeof(h.free_space));
        r(&h.next_page, sizeof(h.next_page));
        r(&h.prev_page, sizeof(h.prev_page));
        r(&h.page_lsn, sizeof(h.page_lsn));
        r(h.reserved, sizeof(h.reserved));
        return h;
    }
};

class DiskManager {
    std::string filepath_;
    std::fstream file_;
    size_t page_size_;
    std::atomic<PageId> next_page_id_{0};
    std::mutex mtx_;

public:
    DiskManager() : page_size_(DEFAULT_PAGE_SIZE) {}

    explicit DiskManager(const std::string& path, size_t page_size = DEFAULT_PAGE_SIZE)
        : filepath_(path), page_size_(page_size) {
        file_.open(filepath_,
            std::ios::in | std::ios::out | std::ios::binary);
        if (!file_.is_open()) {
            file_.open(filepath_,
                std::ios::in | std::ios::out | std::ios::binary | std::ios::trunc);
        }
        if (file_.is_open()) {
            file_.seekg(0, std::ios::end);
            auto file_size = file_.tellg();
            if (file_size > 0) {
                next_page_id_ = static_cast<PageId>(file_size) / page_size_;
            }
        }
    }

    ~DiskManager() {
        if (file_.is_open()) file_.close();
    }

    DiskManager(DiskManager&&) = delete;
    DiskManager& operator=(DiskManager&&) = delete;

    PageId allocate_page() {
        return next_page_id_.fetch_add(1);
    }

    void write_page(PageId pid, const DiskPage& page) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (!file_.is_open()) return;
        auto offset = static_cast<std::streamoff>(pid * page_size_);
        file_.seekp(offset);
        file_.write(reinterpret_cast<const char*>(page.data.data()),
                    static_cast<std::streamsize>(page_size_));
        file_.flush();
    }

    DiskPage read_page(PageId pid) {
        std::lock_guard<std::mutex> lock(mtx_);
        DiskPage page(page_size_);
        page.page_id = pid;
        if (!file_.is_open()) return page;
        auto offset = static_cast<std::streamoff>(pid * page_size_);
        file_.seekg(offset);
        file_.read(reinterpret_cast<char*>(page.data.data()),
                   static_cast<std::streamsize>(page_size_));
        return page;
    }

    size_t page_size() const { return page_size_; }
    PageId page_count() const { return next_page_id_.load(); }
};

class BufferPool {
    size_t capacity_;
    size_t page_size_;
    DiskManager* disk_mgr_;

    struct Frame {
        DiskPage page;
        bool valid = false;
    };

    std::unordered_map<PageId, std::list<PageId>::iterator> page_table_;
    std::list<PageId> lru_list_;
    std::unordered_map<PageId, Frame> frames_;
    std::mutex mtx_;

public:
    BufferPool() : capacity_(DEFAULT_BUFFER_POOL_SIZE),
                   page_size_(DEFAULT_PAGE_SIZE), disk_mgr_(nullptr) {}

    BufferPool(size_t capacity, DiskManager* disk_mgr)
        : capacity_(capacity), page_size_(disk_mgr->page_size()),
          disk_mgr_(disk_mgr) {}

    DiskPage* fetch_page(PageId pid) {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = page_table_.find(pid);
        if (it != page_table_.end()) {
            lru_list_.erase(it->second);
            lru_list_.push_front(pid);
            it->second = lru_list_.begin();
            frames_[pid].page.pin_count++;
            return &frames_[pid].page;
        }

        if (frames_.size() >= capacity_) evict_one();

        Frame frame;
        if (disk_mgr_) {
            frame.page = disk_mgr_->read_page(pid);
        } else {
            frame.page = DiskPage(page_size_);
            frame.page.page_id = pid;
        }
        frame.page.pin_count = 1;
        frame.valid = true;

        lru_list_.push_front(pid);
        page_table_[pid] = lru_list_.begin();
        frames_[pid] = std::move(frame);
        return &frames_[pid].page;
    }

    void unpin_page(PageId pid, bool dirty = false) {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = frames_.find(pid);
        if (it == frames_.end()) return;
        if (it->second.page.pin_count > 0) it->second.page.pin_count--;
        if (dirty) it->second.page.dirty = true;
    }

    DiskPage* new_page() {
        if (!disk_mgr_) return nullptr;
        PageId pid = disk_mgr_->allocate_page();
        auto* page = fetch_page(pid);
        if (page) {
            page->page_id = pid;
            page->dirty = true;
            std::memset(page->data.data(), 0, page->data.size());
        }
        return page;
    }

    void flush_page(PageId pid) {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = frames_.find(pid);
        if (it == frames_.end() || !it->second.page.dirty) return;
        if (disk_mgr_) disk_mgr_->write_page(pid, it->second.page);
        it->second.page.dirty = false;
    }

    void flush_all() {
        std::lock_guard<std::mutex> lock(mtx_);
        for (auto& [pid, frame] : frames_) {
            if (frame.page.dirty && disk_mgr_) {
                disk_mgr_->write_page(pid, frame.page);
                frame.page.dirty = false;
            }
        }
    }

    std::vector<PageId> dirty_pages() const {
        std::vector<PageId> result;
        for (auto& [pid, frame] : frames_) {
            if (frame.page.dirty) result.push_back(pid);
        }
        return result;
    }

    size_t size() const { return frames_.size(); }
    size_t capacity() const { return capacity_; }

private:
    void evict_one() {
        for (auto it = lru_list_.rbegin(); it != lru_list_.rend(); ++it) {
            PageId pid = *it;
            auto& frame = frames_[pid];
            if (frame.page.pin_count == 0) {
                if (frame.page.dirty && disk_mgr_)
                    disk_mgr_->write_page(pid, frame.page);
                page_table_.erase(pid);
                lru_list_.erase(std::next(it).base());
                frames_.erase(pid);
                return;
            }
        }
    }
};

}  // namespace hptree
