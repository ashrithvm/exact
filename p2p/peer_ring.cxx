#include "peer_ring.hxx"

#include <algorithm>
#include <stdexcept>

PeerRing::PeerRing(int32_t _num_peers, int32_t _num_islands)
    : num_peers(_num_peers), num_islands(_num_islands), active_peers(_num_peers, true) {
    build_island_mapping();
}

void PeerRing::build_island_mapping() {
    island_to_peer.clear();

    // Collect active peer IDs
    vector<int32_t> active_ids;
    for (int32_t i = 0; i < num_peers; i++) {
        if (active_peers[i]) {
            active_ids.push_back(i);
        }
    }

    if (active_ids.empty()) {
        return;
    }

    // Modular assignment: island i -> active_ids[i % num_active]
    int32_t num_active = (int32_t)active_ids.size();
    for (int32_t i = 0; i < num_islands; i++) {
        island_to_peer[i] = active_ids[i % num_active];
    }
}

int32_t PeerRing::get_owner(int32_t global_island_id) const {
    auto it = island_to_peer.find(global_island_id);
    if (it == island_to_peer.end()) {
        return -1;
    }
    return it->second;
}

vector<int32_t> PeerRing::get_owned_islands(int32_t peer_id) const {
    vector<int32_t> owned;
    for (auto& pair : island_to_peer) {
        if (pair.second == peer_id) {
            owned.push_back(pair.first);
        }
    }
    return owned;
}

int32_t PeerRing::get_successor(int32_t peer_id) const {
    // Walk forward on the ring until we find an active peer
    for (int32_t offset = 1; offset < num_peers; offset++) {
        int32_t candidate = (peer_id + offset) % num_peers;
        if (active_peers[candidate]) {
            return candidate;
        }
    }
    // Only peer left is itself
    return peer_id;
}

int32_t PeerRing::get_predecessor(int32_t peer_id) const {
    for (int32_t offset = 1; offset < num_peers; offset++) {
        int32_t candidate = (peer_id - offset + num_peers) % num_peers;
        if (active_peers[candidate]) {
            return candidate;
        }
    }
    return peer_id;
}

vector<int32_t> PeerRing::get_k_successors(int32_t peer_id, int32_t k) const {
    vector<int32_t> successors;
    int32_t current = peer_id;
    for (int32_t i = 0; i < k; i++) {
        current = get_successor(current);
        if (current == peer_id) {
            break;  // wrapped around, no more distinct peers
        }
        successors.push_back(current);
    }
    return successors;
}

void PeerRing::remove_peer(int32_t peer_id) {
    if (peer_id < 0 || peer_id >= num_peers || !active_peers[peer_id]) {
        return;
    }

    active_peers[peer_id] = false;

    // Rebuild island mapping to redistribute failed peer's islands
    build_island_mapping();
}

bool PeerRing::is_active(int32_t peer_id) const {
    if (peer_id < 0 || peer_id >= num_peers) {
        return false;
    }
    return active_peers[peer_id];
}

int32_t PeerRing::get_num_active_peers() const {
    int32_t count = 0;
    for (bool active : active_peers) {
        if (active) count++;
    }
    return count;
}

vector<int32_t> PeerRing::get_all_active_peers() const {
    vector<int32_t> result;
    for (int32_t i = 0; i < num_peers; i++) {
        if (active_peers[i]) {
            result.push_back(i);
        }
    }
    return result;
}

int32_t PeerRing::get_num_peers() const {
    return num_peers;
}

int32_t PeerRing::get_num_islands() const {
    return num_islands;
}
