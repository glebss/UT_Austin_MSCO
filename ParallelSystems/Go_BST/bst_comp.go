package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

type Tree struct {
	root *Node
}

type Node struct {
	value int
	left  *Node
	right *Node
}

func (t *Tree) insert(data int) {
	if t.root == nil {
		t.root = &Node{value: data}
	} else {
		t.root.insert(data)
	}
}

func (n *Node) insert(data int) {
	if data <= n.value {
		if n.left == nil {
			n.left = &Node{value: data}
		} else {
			n.left.insert(data)
		}
	} else {
		if n.right == nil {
			n.right = &Node{value: data}
		} else {
			n.right.insert(data)
		}
	}
}

func (t *Tree) in_order_traversal() (vals []int) {
	vals = make([]int, 0)
	if t.root != nil {
		t.root.in_order_traversal(&vals)
	}
	return vals
}

func (n *Node) in_order_traversal(vals *[]int) {
	if n == nil {
		return
	}

	n.left.in_order_traversal(vals)
	*vals = append(*vals, n.value)
	n.right.in_order_traversal(vals)
}

func compute_hash(t *Tree) (hash int) {
	hash = 1
	new_value := 0
	for _, value := range t.in_order_traversal() {
		new_value = value + 2
		hash = (hash*new_value + new_value) % 1000
	}
	return hash
}

func compute_hashes_chunk(trees *[]Tree) (hashes []int) {
	hashes = make([]int, len(*trees))
	for i := 0; i < len(hashes); i++ {
		t := (*trees)[i]
		hashes[i] = compute_hash(&t)
	}
	return hashes
}

type chunk_result_t struct {
	start_idx int
	hashes    []int
}

type ids_pair_t struct {
	id1 int
	id2 int
}

func compare_trees(t1 *Tree, t2 *Tree) bool {
	values1 := t1.in_order_traversal()
	values2 := t2.in_order_traversal()
	if len(values1) != len(values2) {
		return false
	}
	for i := 0; i < len(values1); i++ {
		if values1[i] != values2[i] {
			return false
		}
	}
	return true
}

func arrange_hashes_worker(result_hashes_chan <-chan chunk_result_t, wg *sync.WaitGroup, hash2id *map[int][]int) {
	defer wg.Done()
	for result := range result_hashes_chan {
		start := result.start_idx
		for i, res_hash := range result.hashes {
			id := start + i
			_, ok := (*hash2id)[res_hash]
			if ok {
				(*hash2id)[res_hash] = append((*hash2id)[res_hash], id)
			} else {
				(*hash2id)[res_hash] = []int{id}
			}
		}
	}
}

func calc_hashes_worker(result_hashes_chan chan<- chunk_result_t, i int, chunk_size int, num_workers int, bst_slice *[]Tree, wg *sync.WaitGroup) {
	defer wg.Done()
	start := i * chunk_size
	end := (i + 1) * chunk_size
	if i == num_workers-1 {
		end = len(*bst_slice)
	}
	chunk := (*bst_slice)[start:end]
	result_hashes_chan <- chunk_result_t{hashes: compute_hashes_chunk(&chunk), start_idx: start}
}

func comp_trees_worker(work_channel <-chan ids_pair_t, wg *sync.WaitGroup, trees *[]Tree, mat *[][]bool) {
	defer wg.Done()
	for ids_pair := range work_channel {
		id1 := ids_pair.id1
		id2 := ids_pair.id2
		tree1 := (*trees)[id1]
		tree2 := (*trees)[id2]
		is_equal := compare_trees(&tree1, &tree2)
		(*mat)[id1][id2] = is_equal
		(*mat)[id2][id1] = is_equal
	}
}

func main() {
	// read command line args
	hash_workers_ptr := flag.Int("hash-workers", -1, "Number of hash workers")
	data_workers_ptr := flag.Int("data-workers", -1, "Number of data workers")
	comp_workers_ptr := flag.Int("comp-workers", -1, "Number of comp workers")
	input_file_ptr := flag.String("input", "", "Input file")

	flag.Parse()
	if *hash_workers_ptr == -1 && *data_workers_ptr == -1 && *comp_workers_ptr == -1 {
		log.Println("None of the number of workers specified!")
		os.Exit(1)
	}

	// create array of BSTs
	bst_slice := make([]Tree, 0)

	// read line inputs and create BST for each line
	file, err := os.Open(*input_file_ptr)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		string := scanner.Text()
		nodes := strings.Fields(string)
		var t Tree
		for _, n := range nodes {
			n_val, _ := strconv.Atoi(n)
			t.insert(n_val)
		}
		bst_slice = append(bst_slice, t)
	}

	// generate hash for each BST
	hashes := make([]int, len(bst_slice))
	hash2id := make(map[int][]int)
	if *hash_workers_ptr == 1 && *data_workers_ptr == 1 {
		start_hash := time.Now()
		for i, tree := range bst_slice {
			hashes[i] = compute_hash(&tree)
		}
		elapsed_hash := time.Since(start_hash)
		seconds_hash := elapsed_hash.Seconds()
		fmt.Printf("hashTime: %f\n", seconds_hash)
		for id, hash := range hashes {
			_, ok := hash2id[hash]
			if ok {
				hash2id[hash] = append(hash2id[hash], id)
			} else {
				hash2id[hash] = []int{id}
			}
		}
		elapsed_hash_group := time.Since(start_hash)
		seconds_hash_group := elapsed_hash_group.Seconds()
		fmt.Printf("hashGroupTime: %f\n", seconds_hash_group)

	} else if *hash_workers_ptr > 1 && *data_workers_ptr == 1 {
		start_hash := time.Now()
		var wg sync.WaitGroup
		num_workers := min(*hash_workers_ptr, len(bst_slice))
		chunk_size := len(bst_slice) / num_workers
		wg.Add(num_workers)
		result_hashes_chan := make(chan chunk_result_t, num_workers)
		for i := 0; i < num_workers; i++ {
			go calc_hashes_worker(result_hashes_chan, i, chunk_size, num_workers, &bst_slice, &wg)
		}
		wg.Wait()
		close(result_hashes_chan)
		elapsed_hash := time.Since(start_hash)
		seconds_hash := elapsed_hash.Seconds()
		fmt.Printf("hashTime: %f\n", seconds_hash)
		wg.Add(1)
		go arrange_hashes_worker(result_hashes_chan, &wg, &hash2id)
		wg.Wait()
		elapsed_hash_group := time.Since(start_hash)
		seconds_hash_group := elapsed_hash_group.Seconds()
		fmt.Printf("hashGroupTime: %f\n", seconds_hash_group)
	} else if *hash_workers_ptr > 1 && *data_workers_ptr > 1 {
		if *hash_workers_ptr != *data_workers_ptr {
			log.Println("Different numbers for hash-workers and data-workers not implemented!")
			os.Exit(1)
		}
		start_hash := time.Now()
		var wg sync.WaitGroup
		num_workers := min(*hash_workers_ptr, len(bst_slice))
		wg.Add(num_workers)
		chunk_size := len(bst_slice) / num_workers
		var mutex sync.Mutex
		for i := 0; i < num_workers; i++ {
			go func(i int) {
				start := i * chunk_size
				end := (i + 1) * chunk_size
				if i == num_workers-1 {
					end = len(bst_slice)
				}
				chunk := bst_slice[start:end]
				hashes_chunk := compute_hashes_chunk(&chunk)
				for j := 0; j < end-start; j++ {
					mutex.Lock()
					id := start + j
					res_hash := hashes_chunk[j]
					_, ok := hash2id[res_hash]
					if ok {
						hash2id[res_hash] = append(hash2id[res_hash], id)
					} else {
						hash2id[res_hash] = []int{id}
					}
					mutex.Unlock()
				}
				wg.Done()
			}(i)
		}
		wg.Wait()
		elapsed_hash := time.Since(start_hash)
		seconds_hash := elapsed_hash.Seconds()
		fmt.Printf("hashTime: %f\n", seconds_hash)
		fmt.Printf("hashGroupTime: %f\n", seconds_hash)
	} else if *hash_workers_ptr == 1 && *data_workers_ptr == -1 && *comp_workers_ptr == -1 {
		start_hash := time.Now()
		var wg sync.WaitGroup
		num_workers := min(*hash_workers_ptr, len(bst_slice))
		chunk_size := len(bst_slice) / num_workers
		wg.Add(num_workers)
		result_hashes_chan := make(chan chunk_result_t, num_workers)
		for i := 0; i < num_workers; i++ {
			go calc_hashes_worker(result_hashes_chan, i, chunk_size, num_workers, &bst_slice, &wg)
		}
		wg.Wait()
		close(result_hashes_chan)
		elapsed_hash := time.Since(start_hash)
		seconds_hash := elapsed_hash.Seconds()
		fmt.Printf("hashTime: %f\n", seconds_hash)
	}

	cur_hash_id := 0
	for _, ids := range hash2id {
		if len(ids) == 1 {
			continue
		}
		fmt.Printf("hash%d: ", cur_hash_id)
		for _, id := range ids {
			fmt.Printf("id%d%d ", cur_hash_id, id)
		}
		fmt.Println("")
		cur_hash_id += 1
	}

	if *comp_workers_ptr == -1 {
		return
	}

	// compare trees with identical hashes
	num_trees := len(bst_slice)
	adjacency_matrix := make([][]bool, num_trees)
	for i := range adjacency_matrix {
		adjacency_matrix[i] = make([]bool, num_trees)
		for j := 0; j < num_trees; j++ {
			adjacency_matrix[i][j] = i == j
		}
	}
	start_compare := time.Now()
	if *comp_workers_ptr == 1 {
		var wg sync.WaitGroup
		wg.Add(1)
		go func() {
			defer wg.Done()
			for _, ids := range hash2id {
				if len(ids) == 1 {
					continue
				}
				for idx1, id1 := range ids {
					for _, id2 := range ids[idx1:] {
						tree1 := bst_slice[id1]
						tree2 := bst_slice[id2]
						is_equal := compare_trees(&tree1, &tree2)
						adjacency_matrix[id1][id2] = is_equal
						adjacency_matrix[id2][id1] = is_equal
					}
				}
			}
		}()
		wg.Wait()
	} else {
		num_comp_workers := *comp_workers_ptr
		work_channel := make(chan ids_pair_t, num_comp_workers)
		var wg sync.WaitGroup
		for i := 0; i < num_comp_workers; i++ {
			wg.Add(1)
			go comp_trees_worker(work_channel, &wg, &bst_slice, &adjacency_matrix)
		}

		for _, ids := range hash2id {
			if len(ids) == 1 {
				continue
			}
			for idx1, id1 := range ids {
				for _, id2 := range ids[idx1:] {
					ids_pair := ids_pair_t{id1, id2}
					work_channel <- ids_pair
				}
			}
		}
		close(work_channel)
		wg.Wait()
	}

	elapsed_compare := time.Since(start_compare)
	seconds_compare := elapsed_compare.Seconds()
	fmt.Printf("compareTreeTime: %f\n", seconds_compare)

	// Traverse adjacency matrix and identify groups
	id2group := make(map[int]int)
	cur_group := 0
	for id1 := 0; id1 < num_trees; id1++ {
		_, ok := id2group[id1]
		if ok {
			continue
		}
		cur_group_ids := make([]int, 0)
		cur_group_ids = append(cur_group_ids, id1)
		id2group[id1] = cur_group
		for id2 := 0; id2 < num_trees; id2++ {
			if id1 == id2 {
				continue
			}
			if adjacency_matrix[id1][id2] {
				cur_group_ids = append(cur_group_ids, id2)
				id2group[id2] = cur_group
			}
		}
		if len(cur_group_ids) > 1 {
			fmt.Printf("group %d: ", cur_group)
			for _, id := range cur_group_ids {
				fmt.Printf("id%d%d ", cur_group, id)
			}
			fmt.Println("")
			cur_group += 1
		}
	}

}
