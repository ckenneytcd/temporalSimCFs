import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from src.temp_sim.metrics.replay_buffer import ReplayBuffer


class GraphEstimator:

    def __init__(self, fact, env_model, buffer_path, graph_path):
        self.fact = fact
        self.env_model = env_model
        self.buffer_path = buffer_path
        self.graph_path = graph_path

        self.n_steps = 10

        self.setup()

    def setup(self):
        # Fill replay buffer and generate graph
        self.replay_buffer = self.fill_replay_buffer(self.fact)
        self.G = self.generate_graph()

        # get all shortest paths from between nodes
        print('Generating shortest paths...')
        self.paths = dict(nx.all_pairs_shortest_path(self.G))

    def fill_replay_buffer(self, fact):
        # Generate dataset around fact from env_model
        replay_buffer = ReplayBuffer(capacity=1000)
        try:
            replay_buffer.load(self.buffer_path)
            print('Loaded replay buffer from: {}'.format(self.buffer_path))
        except FileNotFoundError:
            print('Generating data around the factual instance')
            for i in range(1000):
                self.env_model.set_state(fact)
                done = False
                step = 0
                curr_state = fact
                while not done and (step <= self.n_steps) and not replay_buffer.is_full():
                    rand_action = self.env_model.sample_action()
                    print('Rand_action: {}'.format(rand_action))
                    next_state, rew, done, _ = self.env_model.step(rand_action)
                    replay_buffer.add(curr_state, next_state)
                    curr_state = next_state
                    step += 1

            print('Finished. Generated {} instances.'.format(replay_buffer.count))
            replay_buffer.save(self.buffer_path)
            print('Saved buffer at: {}'.format(self.buffer_path))

        return replay_buffer

    def generate_graph(self):
        # Generate graph from dataset
        G = nx.Graph()

        try:
            # load graph if it exists
            G = nx.read_gpickle(self.graph_path)
            print('Loaded graph from: {}'.format(self.graph_path))
        except FileNotFoundError:
            # add nodes and edges
            node_list = self.replay_buffer.get_state_ids()
            edge_list = self.replay_buffer.get_edges()

            G.add_nodes_from(node_list)
            G.add_edges_from(edge_list)
            print('Built graph with {} nodes and {} edges.'.format(len(G.nodes), len(G.edges)))

            nx.write_gpickle(G, self.graph_path)
            print('Saved graph at: {}'.format(self.graph_path))

        # plot graph if small enough
        if len(G.nodes) < 100:
            nx.draw(G, with_labels=True, font_weight='bold')
            plt.show()

        return G

    def get_shortest_path(self, f, cf):
        # Get shortest path from f to cf
        f_id = self.replay_buffer.get_id_of_state(f)
        cf_id = self.replay_buffer.get_id_of_state(cf)

        if f_id == -1 or cf_id == -1:
            return +100

        try:
            # if cf_id is in the list of shortest paths for f_id
            shortest_path = len(self.paths[f_id][cf_id])
        except KeyError:
            shortest_path = np.inf

        return shortest_path