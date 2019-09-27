from __future__ import absolute_import
from collections import deque, namedtuple
import warnings
import random
import queue as Q
import numpy as np


# This is to be understood as a transition: Given `state0`, performing `action`
# yields `reward` and results in `state1`, which might be `terminal`.
Experience = namedtuple('Experience', 'state0, action, reward, state1, terminal1')


def sample_batch_indexes(low, high, size):
    """Return a sample of (size) unique elements between low and high

        # Argument
            low (int): The minimum value for our samples
            high (int): The maximum value for our samples
            size (int): The number of samples to pick

        # Returns
            A list of samples of length size, with values between low and high
        """
    if high - low >= size:
        # We have enough data. Draw without replacement, that is each index is unique in the
        # batch. We cannot use `np.random.choice` here because it is horribly inefficient as
        # the memory grows. See https://github.com/numpy/numpy/issues/2764 for a discussion.
        # `random.sample` does the same thing (drawing without replacement) and is way faster.
        try:
            r = xrange(low, high)
        except NameError:
            r = range(low, high)
        batch_idxs = random.sample(r, size)
    else:
        # Not enough data. Help ourselves with sampling from the range, but the same index
        # can occur multiple times. This is not good and should be avoided by picking a
        # large enough warm-up phase.
        warnings.warn('Not enough entries to sample without replacement. Consider increasing your warm-up phase to avoid oversampling!')
        batch_idxs = np.random.random_integers(low, high - 1, size=size)
    assert len(batch_idxs) == size
    return batch_idxs


class RingBuffer(object):
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.data = deque(maxlen=maxlen)

    def __len__(self):
        return self.length()

    def __getitem__(self, idx):
        """Return element of buffer at specific index

        # Argument
            idx (int): Index wanted

        # Returns
            The element of buffer at given index
        """
        if idx < 0 or idx >= self.length():
            raise KeyError()
        return self.data[idx]

    def append(self, v):
        """Append an element to the buffer

        # Argument
            v (object): Element to append
        """
        self.data.append(v)

    def length(self):
        """Return the length of Deque

        # Argument
            None

        # Returns
            The lenght of deque element
        """
        return len(self.data)

def zeroed_observation(observation):
    """Return an array of zeros with same shape as given observation

    # Argument
        observation (list): List of observation
    
    # Return
        A np.ndarray of zeros with observation.shape
    """
    if hasattr(observation, 'shape'):
        return np.zeros(observation.shape)
    elif hasattr(observation, '__iter__'):
        out = []
        for x in observation:
            out.append(zeroed_observation(x))
        return out
    else:
        return 0.

class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    Story data with its priority in the tree.
    """
    #data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        #self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, index):
        tree_idx = index + self.capacity - 1
        #self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        #self.data_pointer += 1
        #if self.data_pointer >= self.capacity:  # replace when exceed the capacity
        #    self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx] #, self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class MemoryTD(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 100.  # clipped abs error

    def __init__(self, capacity, pri_reservoir_retention, pri_td_retention):
        self.tree = SumTree(capacity)
        self.min_prob = 100
    def store(self, index):
        #max_p = np.max(self.tree.tree[-self.tree.capacity:])
        #if max_p == 0:
        max_p = self.abs_err_upper
        self.tree.add(max_p, index)   # set the max p for new p

    def sample(self, n):
        b_idx, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n))
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        #min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i] = np.power(prob/self.min_prob, -self.beta)
            b_idx[i] = idx
        return b_idx, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)
            self.min_prob = min(self.min_prob, p)


class Memory(object):
    def __init__(self, window_length, ignore_episode_boundaries=False):
        self.window_length = window_length
        self.ignore_episode_boundaries = ignore_episode_boundaries

        self.recent_observations = deque(maxlen=window_length)
        self.recent_terminals = deque(maxlen=window_length)

        self.recent_ob = deque(maxlen=window_length)
        self.recent_ter = deque(maxlen=window_length)
        self.recent_act = deque(maxlen=window_length)
        self.recent_re = deque(maxlen=window_length)

    def sample(self, batch_size, batch_idxs=None):
        raise NotImplementedError()

    def append(self, observation, action, reward, terminal, training=True):
        self.recent_observations.append(observation)
        self.recent_terminals.append(terminal)

    def get_recent_state(self, current_observation):
        """Return list of last observations

        # Argument
            current_observation (object): Last observation

        # Returns
            A list of the last observations
        """
        # This code is slightly complicated by the fact that subsequent observations might be
        # from different episodes. We ensure that an experience never spans multiple episodes.
        # This is probably not that important in practice but it seems cleaner.
        state = [current_observation]
        idx = len(self.recent_observations) - 1
        for offset in range(0, self.window_length - 1):
            current_idx = idx - offset
            current_terminal = self.recent_terminals[current_idx - 1] if current_idx - 1 >= 0 else False
            if current_idx < 0 or (not self.ignore_episode_boundaries and current_terminal):
                # The previously handled observation was terminal, don't add the current one.
                # Otherwise we would leak into a different episode.
                break
            state.insert(0, self.recent_observations[current_idx])
        while len(state) < self.window_length:
            state.insert(0, zeroed_observation(state[0]))
        return state

    def get_config(self):
        """Return configuration (window_length, ignore_episode_boundaries) for Memory
        
        # Return
            A dict with keys window_length and ignore_episode_boundaries
        """
        config = {
            'window_length': self.window_length,
            'ignore_episode_boundaries': self.ignore_episode_boundaries,
        }
        return config

class SequentialMemory(Memory):
    def __init__(self, limit, **kwargs):
        super(SequentialMemory, self).__init__(**kwargs)
        
        self.prioritized = True
        self.important_sampling = False
        self.reservoir_retention = False
        self.td_retention = True

        self.limit = limit
        self.memory_size = limit
        self.TD_MAX = 100
        self.memory_counter = 0
        # Do not use deque to implement the memory. This data structure may seem convenient but
        # it is way too slow on random access. Instead, we use our own ring buffer implementation.
        self.actions = RingBuffer(limit)
        self.rewards = RingBuffer(limit)
        self.terminals = RingBuffer(limit)
        self.observations = RingBuffer(limit)

        self.memorytd = MemoryTD(capacity=self.memory_size,
                                 pri_reservoir_retention=self.reservoir_retention,
                                 pri_td_retention=self.td_retention)
        
        self.memoryexp = []
        self.Td = []

        self.que = Q.PriorityQueue()

    def batch_update(self, sample_index, abs_errors):
        #clipped_errors = np.minimum(self.Td, 1)
        
        for i, p in zip(sample_index, abs_errors):
            self.Td[i] = p
            self.que.put((p,i))
        
        tree_idx = sample_index + (self.memory_size - 1)
        self.memorytd.batch_update(tree_idx, abs_errors)

    def sample(self, batch_size, batch_idxs=None):
        """Return a randomized batch of experiences

        # Argument
            batch_size (int): Size of the all batch
            batch_idxs (int): Indexes to extract
        # Returns
            A list of experiences randomly selected
        """
        # It is not possible to tell whether the first state in the memory is terminal, because it
        # would require access to the "terminal" flag associated to the previous state. As a result
        # we will never return this first state (only using `self.terminals[0]` to know whether the
        # second state is terminal).
        # In addition we need enough entries to fill the desired window length.
        # assert self.nb_entries >= self.window_length + 2, 'not enough entries in the memory'

        # if batch_idxs is None:
        #     # Draw random indexes such that we have enough entries before each index to fill the
        #     # desired window length.
        #     batch_idxs = sample_batch_indexes(
        #         self.window_length, self.nb_entries - 1, size=batch_size)
        # batch_idxs = np.array(batch_idxs) + 1
        # assert np.min(batch_idxs) >= self.window_length + 1
        # assert np.max(batch_idxs) < self.nb_entries
        # assert len(batch_idxs) == batch_size

        # # Create experiences
        # experiences = []
        # for idx in batch_idxs:
        #     terminal0 = self.terminals[idx - 2]
        #     while terminal0:
        #         # Skip this transition because the environment was reset here. Select a new, random
        #         # transition and use this instead. This may cause the batch to contain the same
        #         # transition twice.
        #         idx = sample_batch_indexes(self.window_length + 1, self.nb_entries, size=1)[0]
        #         terminal0 = self.terminals[idx - 2]
        #     assert self.window_length + 1 <= idx < self.nb_entries

        #     # This code is slightly complicated by the fact that subsequent observations might be
        #     # from different episodes. We ensure that an experience never spans multiple episodes.
        #     # This is probably not that important in practice but it seems cleaner.
        #     state0 = [self.observations[idx - 1]]
        #     for offset in range(0, self.window_length - 1):
        #         current_idx = idx - 2 - offset
        #         assert current_idx >= 1
        #         current_terminal = self.terminals[current_idx - 1]
        #         if current_terminal and not self.ignore_episode_boundaries:
        #             # The previously handled observation was terminal, don't add the current one.
        #             # Otherwise we would leak into a different episode.
        #             break
        #         state0.insert(0, self.observations[current_idx])
        #     while len(state0) < self.window_length:
        #         state0.insert(0, zeroed_observation(state0[0]))
        #     action = self.actions[idx - 1]
        #     reward = self.rewards[idx - 1]
        #     terminal1 = self.terminals[idx - 1]

        #     # Okay, now we need to create the follow-up state. This is state0 shifted on timestep
        #     # to the right. Again, we need to be careful to not include an observation from the next
        #     # episode if the last state is terminal.
        #     state1 = [np.copy(x) for x in state0[1:]]
        #     state1.append(self.observations[idx])

        #     assert len(state0) == self.window_length
        #     assert len(state1) == len(state0)
        #     experiences.append(Experience(state0=state0, action=action, reward=reward,
        #                                   state1=state1, terminal1=terminal1))
        # assert len(experiences) == batch_size

        if self.prioritized :
            tree_idx, ISWeights = self.memorytd.sample(batch_size)
            #print("1",ISWeights)
            sample_index = tree_idx - (self.memory_size - 1)
        else:
            # 随机取出记忆
            if self.memory_counter > self.memory_size:
                sample_index = np.random.choice(self.memory_size, size=batch_size)
            else:
                sample_index = np.random.choice(self.memory_counter, size=batch_size)

        #print("!:", sample_index)
        #print("!:", ISWeights)
        experiences = [self.memoryexp[sample_index[i]] for i in range(batch_size)] #np.array(self.memoryexp)[sample_index] 
        assert len(experiences) == batch_size
        if self.important_sampling:
            return experiences, ISWeights, sample_index
        else:
            return experiences, np.array([1 for i in range(batch_size)]), sample_index

    def append(self, observation, action, reward, terminal, training=True):
        """Append an observation to the memory

        # Argument
            observation (dict): Observation returned by environment
            action (int): Action taken to obtain this observation
            reward (float): Reward obtained by taking this action
            terminal (boolean): Is the state terminal
        """ 
        super(SequentialMemory, self).append(observation, action, reward, terminal, training=training)
        
        # This needs to be understood as follows: in `observation`, take `action`, obtain `reward`
        # and weather the next state is `terminal` or not.
        # if training:
        #     self.observations.append(observation)
        #     self.actions.append(action)
        #     self.rewards.append(reward)
        #     self.terminals.append(terminal)
        
        ###
        if len(self.recent_ob) == self.window_length:
            state0 = []
            state1 = []
            for id in range(self.window_length):
                state0.append(self.recent_ob[id])
                if id != 0:
                    state1.append(self.recent_ob[id])
            
            state1.append(observation)
            idx = self.window_length-1
            tpexp = Experience(state0=state0, action=self.recent_act[idx], reward=self.recent_re[idx],
                               state1=state1, terminal1=self.recent_ter[idx])

            if self.memory_counter < self.memory_size:
                self.memoryexp.append(tpexp)
                self.Td.append(self.TD_MAX)
                self.memorytd.store(self.memory_counter)
                index = -1

                if(self.td_retention):
                    self.que.put((self.TD_MAX,self.memory_counter))

            elif self.td_retention:
                #index1 = np.argmin(self.Td)
                cur = self.que.get()
                while(self.Td[cur[1]] != cur[0]):
                    #self.que.put((self.Td[cur[1]],cur[1]))
                    cur = self.que.get()
                
                index = cur[1]
                #print(str(index)+" : "+str(index1))
                #print(str(self.Td[index])+" !! "+str(self.Td[index1]))

            elif self.reservoir_retention:
                q = random.randint(0, self.memory_counter)
                if q < self.memory_size:
                    index = random.randint(0, self.memory_size-1)
                else:
                    index = -1 #cancel
            else:
                index = self.memory_counter % self.memory_size        

            if index!=-1:
                if self.prioritized:
                    #transition = np.hstack((s, [a, r], s_))
                    self.memorytd.store(index)
                if self.td_retention:
                    self.que.put((self.TD_MAX, index))
                #print("index",index)
                self.memoryexp[index] = tpexp
                self.Td[index] = self.TD_MAX #check value

            self.memory_counter += 1
            #print(self.memory_counter)

        if terminal:
            self.recent_ob.clear()
            self.recent_act.clear()
            self.recent_re.clear()
            self.recent_ter.clear()
        else:
            self.recent_ob.append(observation)
            self.recent_act.append(action)
            self.recent_re.append(reward)
            self.recent_ter.append(terminal)
        
    @property
    def nb_entries(self):
        """Return number of observations

        # Returns
            Number of observations
        """
        if(self.memory_counter < self.memory_size):
            return self.memory_counter
        else:
            return self.memory_size
        ##return len(self.observations)

    def get_config(self):
        """Return configurations of SequentialMemory

        # Returns
            Dict of config
        """
        config = super(SequentialMemory, self).get_config()
        config['limit'] = self.limit
        return config


class EpisodeParameterMemory(Memory):
    def __init__(self, limit, **kwargs):
        super(EpisodeParameterMemory, self).__init__(**kwargs)
        self.limit = limit

        self.params = RingBuffer(limit)
        self.intermediate_rewards = []
        self.total_rewards = RingBuffer(limit)

    def sample(self, batch_size, batch_idxs=None):
        """Return a randomized batch of params and rewards

        # Argument
            batch_size (int): Size of the all batch
            batch_idxs (int): Indexes to extract
        # Returns
            A list of params randomly selected and a list of associated rewards
        """
        if batch_idxs is None:
            batch_idxs = sample_batch_indexes(0, self.nb_entries, size=batch_size)
        assert len(batch_idxs) == batch_size

        batch_params = []
        batch_total_rewards = []
        for idx in batch_idxs:
            batch_params.append(self.params[idx])
            batch_total_rewards.append(self.total_rewards[idx])
        return batch_params, batch_total_rewards

    def append(self, observation, action, reward, terminal, training=True):
        """Append a reward to the memory

        # Argument
            observation (dict): Observation returned by environment
            action (int): Action taken to obtain this observation
            reward (float): Reward obtained by taking this action
            terminal (boolean): Is the state terminal
        """
        super(EpisodeParameterMemory, self).append(observation, action, reward, terminal, training=training)
        if training:
            self.intermediate_rewards.append(reward)

    def finalize_episode(self, params):
        """Closes the current episode, sums up rewards and stores the parameters

        # Argument
            params (object): Parameters associated with the episode to be stored and then retrieved back in sample()
        """
        total_reward = sum(self.intermediate_rewards)
        self.total_rewards.append(total_reward)
        self.params.append(params)
        self.intermediate_rewards = []

    @property
    def nb_entries(self):
        """Return number of episode rewards

        # Returns
            Number of episode rewards
        """
        return len(self.total_rewards)

    def get_config(self):
        """Return configurations of SequentialMemory

        # Returns
            Dict of config
        """
        config = super(SequentialMemory, self).get_config()
        config['limit'] = self.limit
        return config
