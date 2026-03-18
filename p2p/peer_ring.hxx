#ifndef P2P_PEER_RING_HXX
#define P2P_PEER_RING_HXX

#include <cstdint>
#include <map>
#include <vector>

using std::map;
using std::vector;

class PeerRing {
   private:
    int32_t num_peers;
    int32_t num_islands;
    vector<bool> active_peers;
    map<int32_t, int32_t> island_to_peer;  // global_island_id -> owning peer rank

    void build_island_mapping();

   public:
    PeerRing(int32_t _num_peers, int32_t _num_islands);

    // Returns the owning peer rank for a given global island ID
    int32_t get_owner(int32_t global_island_id) const;

    // Returns all global island IDs owned by the given peer
    vector<int32_t> get_owned_islands(int32_t peer_id) const;

    // Returns the next active peer on the ring
    int32_t get_successor(int32_t peer_id) const;

    // Returns the previous active peer on the ring
    int32_t get_predecessor(int32_t peer_id) const;

    // Returns k successor peers for replication targets
    vector<int32_t> get_k_successors(int32_t peer_id, int32_t k) const;

    // Removes a peer from the ring and reassigns its islands to successors
    void remove_peer(int32_t peer_id);

    // Returns true if the peer is still active
    bool is_active(int32_t peer_id) const;

    // Returns the number of currently active peers
    int32_t get_num_active_peers() const;

    // Returns all active peer ranks
    vector<int32_t> get_all_active_peers() const;

    int32_t get_num_peers() const;
    int32_t get_num_islands() const;
};

#endif
