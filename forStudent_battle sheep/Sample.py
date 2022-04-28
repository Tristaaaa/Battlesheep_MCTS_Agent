import STcpClient
import numpy as np
import random
from gameRule import *
from time import time as clock
from copy import deepcopy
from queue import Queue
from random import choice
import math

#player = 4
#turn = 3
#agent = UctMctsAgent()

class MCTSMeta:
    EXPLORATION = 0.5
    RAVE_CONST = 300
    RANDOMNESS = 0.5
    POOLRAVE_CAPACITY = 10
    K_CONST = 10
    A_CONST = 0.25
    WARMUP_ROLLOUTS = 7
    player = 4

'''
class GameMeta:
    PLAYERS = {'none': 0, 'player1': 1, 'player2': 2, 'player3': 3, 'player4': 4}
    INF = float('inf')
    GAME_OVER = -1
    EDGE1 = 1
    EDGE2 = 2
    #NEIGHBOR_PATTERNS = ((-1, 0), (0, -1), (-1, 1), (0, 1), (1, 0), (1, -1))
    NEIGHBOR_PATTERNS_even = [[-1, -1], [0, -1], [-1, 0], [1, 0], [-1, 1], [0, 1]]
    NEIGHBOR_PATTERNS_odd = [[0, -1], [1, -1], [-1, 0], [1, 0], [0, 1], [1, 1]]
'''

class Node:
    """
    Node for the MCTS. Stores the move applied to reach this node from its parent,
    stats for the associated game position, children, parent and outcome
    (outcome==none unless the position ends the game).
    Args:
        move:
        parent:
        N (int): times this position was visited
        Q (int): average reward (wins-losses) from this position
        Q_RAVE (int): times this move has been critical in a rollout
        N_RAVE (int): times this move has appeared in a rollout
        children (dict): dictionary of successive nodes
        outcome (int): If node is a leaf, then outcome indicates
                       the winner, else None
    """

    def __init__(self, move: tuple = None, num: tuple = None, direc: tuple = None, parent: object = None):
        """
        Initialize a new node with optional move and parent and initially empty
        children list and rollout statistics and unspecified outcome.
        """
        self.move = move
        self.dir = direc
        self.num = num
        self.parent = parent
        self.N = 0  # times this position was visited
        self.Q = 0  # average reward (wins-losses) from this position
        self.Q_RAVE = 0  # times this move has been critical in a rollout
        self.N_RAVE = 0  # times this move has appeared in a rollout
        self.children = {}
        #self.outcome = GameMeta.PLAYERS['None']

    def add_children(self, children) -> None:
        """
        Add a list of nodes to the children of this node.
        """
        for child in children:
            self.children[str([child.move, child.num, child.dir])] = child

    @property
    def value(self, explore: float = MCTSMeta.EXPLORATION):
        """
        Calculate the UCT value of this node relative to its parent, the parameter
        "explore" specifies how much the value should favor nodes that have
        yet to be thoroughly explored versus nodes that seem to have a high win
        rate.
        Currently explore is set to 0.5.
        """
        # if the node is not visited, set the value as infinity. Nodes with no visits are on priority
        # (lambda: print("a"), lambda: print("b"))[test==true]()
        if self.N == 0:
            return 0 if explore == 0 else float('inf')
        else:
            return self.Q / self.N + explore * math.sqrt(2 * math.log(self.parent.N) / self.N)  # exploitation + exploration

class UctMctsAgent:
    """
    Basic no frills implementation of an agent that preforms MCTS for hex.
    Attributes:
        root_state (GameState): Game simulator that helps us to understand the game situation
        root (Node): Root of the tree search
        run_time (int): time per each run
        node_count (int): the whole nodes in tree
        num_rollouts (int): The number of rollouts for each search
        EXPLORATION (int): specifies how much the value should favor
                           nodes that have yet to be thoroughly explored versus nodes
                           that seem to have a high win rate.
    """

    def __init__(self, mapStat, sheepStat):
        self.root_mapstate = deepcopy(mapStat)
        self.root_sheepstate = deepcopy(sheepStat)
        self.root = Node()
        self.run_time = 0
        self.node_count = 0
        self.num_rollouts = 0

    def search(self, time_budget: int) -> None:
        """
        Search and update the search tree for a
        specified amount of time in seconds.
        """
        start_time = clock()
        num_rollouts = 0

        # do until we exceed our time budget
        while clock() - start_time < time_budget:
            node, mapstate, sheepstate = self.select_node()
            outcome = self.roll_out(mapstate, sheepstate)
            self.backup(node, outcome)
            num_rollouts += 1
        run_time = clock() - start_time
        node_count = self.tree_size()
        self.run_time = run_time
        self.node_count = node_count
        self.num_rollouts = num_rollouts

    def select_node(self) -> tuple:
        """
        Select a node in the tree to preform a single simulation from.
        """
        global turn, player

        node = self.root
        mapstate = deepcopy(self.root_mapstate)
        sheepstate = deepcopy(self.root_sheepstate)
        turn = player - 1

        # stop if we find reach a leaf node
        while len(node.children) != 0:
            # descend to the maximum value node, break ties at random
            children = node.children.values()
            max_value = max(children, key=lambda n: n.value).value
            max_nodes = [n for n in node.children.values()
                         if n.value == max_value]
            node = choice(max_nodes)
            if turn + 1 == player:
                if node.dir != -1:
                    #if [node.move[i], node.dir[i]] not in checkRemainMove(tmp + 1, tmp_map, tmp_sheep):
                    #    break
                    sheepstate, mapstate = play(turn + 1, mapstate, sheepstate, [node.move, node.num, node.dir])
                turn = (turn + 1) % 4
            else:
                for i in range(3):
                    if node.dir[i] != -1:
                        sheepstate, mapstate = play(turn + 1, mapstate, sheepstate, [node.move[i], node.num[i], node.dir[i]])
                    turn = (turn + 1) % 4
            # if some child node has not been explored select it before expanding
            # other children
            if node.N == 0:
                return node, mapstate, sheepstate

        # if we reach a leaf node generate its children and return one of them
        # if the node is terminal, just return the terminal node
        if self.expand(node, mapstate, sheepstate):
            A = list(node.children.values())
            node = A[np.random.randint(np.shape(A)[0])]
                #tmp_map = deepcopy(mapstate)
                #tmp_sheep = deepcopy(sheepstate)
            if turn + 1 != player:
                for i in range(3):
                    if node.dir[i] != -1:
                            #if [node.move[i], node.dir[i]] not in checkRemainMove(tmp + 1, tmp_map, tmp_sheep):
                            #    break
                        sheepstate, mapstate = play(turn + 1, mapstate, sheepstate, [node.move[i], node.num[i], node.dir[i]])
                    turn = (turn + 1) % 4
                #if tmp == (turn + 3) % 4:
                #    check = 1 
            else:
                if node.dir != -1:
                    sheepstate, mapstate = play(turn + 1, mapstate, sheepstate, [node.move, node.num, node.dir])
                turn = (turn + 1) % 4
                #if check == 1:
                #    turn = tmp
                #    mapstate = deepcopy(tmp_map)
                #    sheepstate = deepcopy(tmp_sheep)
                #    break
                
        return node, mapstate, sheepstate

    @staticmethod
    def expand(parent: Node, mapStat, sheepStat) -> bool:
        """
        Generate the children of the passed "parent" node based on the available
        moves in the passed gamestate and add them to the tree.
        Returns:
            bool: returns false If node is leaf (the game has ended).
        """
        global turn, player

        children = []
        if end_game_check(mapStat, sheepStat) == True:
            # game is over at this node so nothing to expand
            return False
        if turn + 1 == player:
            if not checkRemainMove(turn + 1, mapStat, sheepStat):
                children.append(Node([-1, -1], -1, -1, parent))
                #children[len(children) - 1].N = -1
                #return False
            else:
                for move in checkRemainMove(turn + 1, mapStat, sheepStat):
                    for i in range(1, int(sheepStat[move[0][0]][move[0][1]])):
                        children.append(Node(move[0], i, move[1], parent))
        else:
            moves = {}
            nums = {}
            direcs = {}
            tmp = turn
            for i in range(3):
                moves[i] = []
                nums[i] = []
                direcs[i] = []
                if not checkRemainMove((tmp + i) % 4 + 1, mapStat, sheepStat):
                    moves[i].append([-1, -1])
                    nums[i].append(-1)
                    direcs[i].append(-1)
                else:
                    for move in checkRemainMove((tmp + i) % 4 + 1, mapStat, sheepStat):
                        for j in range(1, int(sheepStat[move[0][0]][move[0][1]])):
                            moves[i].append(move[0])
                            nums[i].append(j)
                            direcs[i].append(move[1])
            for i in range(len(direcs[0])):
                for j in range(len(direcs[1])):
                    for k in range(len(direcs[2])):
                        children.append(Node([moves[0][i], moves[1][j], moves[2][k]], [nums[0][i], nums[1][j], nums[2][k]], [direcs[0][i], direcs[1][j], direcs[2][k]], parent))
        parent.add_children(children)
        return True

    @staticmethod
    def roll_out(mapStat, sheepStat) -> list:
        """
        Simulate an entirely random game from the passed state and return the winning
        player.
        Args:
            state: game state
        Returns:
            int: winner of the game
        """
        global turn

        tmp = turn
        #moves = checkRemainMove()  # Get a list of all possible moves in current state of the game
        mapstate = mapStat
        sheepstate = sheepStat
        while end_game_check(mapstate, sheepstate) != True:
            moves = checkRemainMove(tmp + 1, mapstate, sheepstate)
            if not moves:
                tmp = (tmp + 1) % 4
                continue
            #print(moves[np.random.randint(np.shape(moves)[0])])
            move = np.array(moves[np.random.randint(np.shape(moves)[0])], dtype=object)
            num = random.randint(1, sheepstate[move[0][0]][move[0][1]])
            sheepstate, mapstate = play(tmp + 1, mapstate, sheepstate, [move[0], num, move[1]])
            tmp = (tmp + 1) % 4
            #moves.remove(move)
        
        score = []
        for player in range(1, 5):
            _, n_field, max_field = getConnectRegion(player, mapStat)
            #print(f'{player}=3*{n_field}+{max_field}')
            s = 3 * n_field + max_field
            score.append(s)
        return score

    @staticmethod
    def backup(node: Node, outcome: list) -> None:
        """
        Update the node statistics on the path from the passed node to root to reflect
        the outcome of a randomly simulated playout.
        Args:
            node:
            turn: winner turn
            outcome: outcome of the rollout
        Returns:
            object:
        """
        # Careful: The reward is calculated for player who just played
        # at the node and not the next player to play
        #reward = 0 if outcome == turn else 1
        global turn, player

        reward = 0
        tmp = turn
        rank = sorted(range(len(outcome)), key=lambda k: outcome[k], reverse = True)
        if tmp + 1 != player:
            if player == rank[0]:
                reward = 5
            elif player == rank[1]:
                reward = 3
            elif player == rank[2]:
                reward = 1
            elif player == rank[3]:
                reward = 0
        else:
            reward = 0

        while node is not None:
            node.N += 1
            node.Q += reward
            node = node.parent

            if tmp + 1 != player:
                if player == rank[0]:
                    reward = 5
                elif player == rank[1]:
                    reward = 3
                elif player == rank[2]:
                    reward = 1
                elif player == rank[3]:
                    reward = 0
                tmp = (tmp - 1) % 4
            else:
                reward = 0
                tmp = (tmp - 3) % 4


    def best_move(self) -> tuple:
        """
        Return the best move according to the current tree. 
        Returns:
            best move in terms of the most simulations number unless the game is over
        """
        if end_game_check(self.root_mapstate, self.root_sheepstate) == True:
            return -1 #GameMeta.GAME_OVER

        # choo se the move of the most simulated node breaking ties randomly
        max_value = max(self.root.children.values(), key=lambda n: n.N).N
        #print(len(self.root.children.values()))
        #print(max_value)
        max_nodes = [n for n in self.root.children.values() if n.N == max_value]
        bestchild = choice(max_nodes)
        return [bestchild.move, bestchild.num, bestchild.dir]

    def move(self, move) -> None:
        """
        Make the passed move and update the tree appropriately. It is
        designed to let the player choose an action manually (which might
        not be the best action).
        Args:
            move:
        """
        global player

        if str(move) in self.root.children:
            child = self.root.children[str(move)]
            child.parent = None
            self.root = child
            sheepstate, mapstate = play(player, self.root_mapstate, self.root_sheepstate, [child.move, child.num, child.dir])
            self.root_mapstate = mapstate
            self.root_sheepstate = sheepstate
            return

        # if for whatever reason the move is not in the children of
        # the root just throw out the tree and start over
        sheepstate, mapstate = play(player, self.root_mapstate, self.root_sheepstate, [child.move, child.num, child.dir])
        self.root = Node()
        self.root_mapstate = mapstate
        self.root_sheepstate = sheepstate

    def set_gamestate(self, mapStat, sheepStat) -> None:
        """
        Set the root_state of the tree to the passed gamestate, this clears all
        the information stored in the tree since none of it applies to the new
        state.
        """
        self.root_mapstate = deepcopy(mapStat)
        self.root_sheepstate = deepcopy(sheepStat)
        self.root = Node()
        #self.node_count = 0

    def statistics(self) -> tuple:
        return self.num_rollouts, self.node_count, self.run_time

    def tree_size(self) -> int:
        """
        Count nodes in tree by BFS.
        """
        Q = Queue()
        count = 0
        Q.put(self.root)
        while not Q.empty():
            node = Q.get()
            count += 1
            for child in node.children.values():
                Q.put(child)
        return count


'''
    選擇起始位置
    選擇範圍僅限場地邊緣(至少一個方向為牆)
    
    return: init_pos
    init_pos=[x,y],代表起始位置
    
'''
def isValid(x, y):
    return (x >= 0) and (x < 12) and (y >= 0) and (y < 12)

def InitPos(mapStat):
    init_pos = [0, 0]
    '''
        Write your code here

    '''
    global agent
    agent = UctMctsAgent(mapStat, mapStat)
    #agent.search(1)
    even = [[-1, -1], [0, -1], [-1, 0], [1, 0], [-1, 1], [0, 1]]
    odd = [[0, -1], [1, -1], [-1, 0], [1, 0], [0, 1], [1, 1]]

    even_sec = [[-1, -2], [0, -2], [1, -2], [-2, -1], [1, -1], [-2, 0], [2, 0], [-2, 1], [1, 1], [-1, 2], [0, 2], [1, 2]]
    odd_sec = [[-1, -2], [0, -2], [1, -2], [-1, -1], [2, -1], [-2, 0], [2, 0], [-1, 1], [2, 1], [-1, 2], [0, 2], [1, 2]]

    while(1):
        init_pos = [random.randint(0, 11), random.randint(0, 11)]
        check = checkValidInit(mapStat, init_pos)
        if check == True:
            cnt = 0
            if init_pos[1] % 2 == 0:
                for i in range(6):
                    tmp1 = init_pos[0] + even[i][0]
                    tmp2 = init_pos[1] + even[i][1]
                    if not isValid(tmp1, tmp2):
                        continue
                    if mapStat[tmp1, tmp2] == 0:
                        cnt += 1
                for i in range(12):
                    tmp1 = init_pos[0] + even_sec[i][0]
                    tmp2 = init_pos[1] + even_sec[i][1]
                    if not isValid(tmp1, tmp2):
                        continue
                    if mapStat[tmp1, tmp2] == 0:
                        cnt += 1
            else:
                for i in range(6):
                    tmp1 =init_pos[0] + odd[i][0]
                    tmp2 = init_pos[1] + odd[i][1]
                    if not isValid(tmp1, tmp2):
                        continue
                    if mapStat[tmp1, tmp2] == 0:
                        cnt += 1
                for i in range(12):
                    tmp1 =init_pos[0] + odd_sec[i][0]
                    tmp2 = init_pos[1] + odd_sec[i][1]
                    if not isValid(tmp1, tmp2):
                        continue
                    if mapStat[tmp1, tmp2] == 0:
                        cnt += 1
            if cnt > 10:
                break
    return init_pos


'''
    產出指令
    
    input: 
    playerID: 你在此局遊戲中的角色(1~4)
    mapStat : 棋盤狀態(list of list), 為 12*12矩陣, 
              0=可移動區域, -1=障礙, 1~4為玩家1~4佔領區域
    sheepStat : 羊群分布狀態, 範圍在0~16, 為 12*12矩陣

    return Step
    Step : 3 elements, [(x,y), m, dir]
            x, y 表示要進行動作的座標 
            m = 要切割成第二群的羊群數量
            dir = 移動方向(1~6),對應方向如下圖所示
              1  2
            3  x  4
              5  6
'''
def GetStep(playerID, mapStat, sheepStat):
    step = [(0, 0), 0, 1]
    '''
    Write your code here
    
    '''
    start = clock()
    global agent
    #agent = UctMctsAgent(mapStat, sheepStat)
    global player
    player = playerID
    global turn
    turn = player - 1

    search_time = 4.0

    '''if agent.node_count == 0:
        agent.search(1)'''

    moves = []
    nums = []
    direcs = []
    if agent.node_count == 0:
        agent.set_gamestate(mapStat, sheepStat)
        search_time = 3.5
    else:
        for i in range(3):
            dest_by = ((mapStat - agent.root_mapstate) == ((playerID + i) % 4 + 1))
            ind_d = np.where(dest_by == 1)
            move_by = ((agent.root_sheepstate - sheepStat) > 0) & (mapStat == ((playerID + i) % 4 + 1))
            ind_m = np.where(move_by == 1)
            if len(ind_d[0]) == 0 or len(ind_m[0]) == 0:
                moves.append([-1, -1])
                nums.append(-1)
                direcs.append(-1)
                continue
            #print([ind_m[0][0], ind_m[1][0]])
            #print([ind_d[0][0], ind_d[1][0]])
            #print(ind_m[0][0], ind_m[1][0])
            moves.append([ind_m[0][0], ind_m[1][0]])
            #print(ind_d[0], ind_d[1])
            nums.append(int(sheepStat[ind_d[0][0]][ind_d[1][0]]))
            if ind_m[1][0] % 2 == 0:
                if (ind_d[0][0] < ind_m[0][0]) & (ind_d[1][0] < ind_m[1][0]):
                    direcs.append(1)
                elif (ind_d[0][0] >= ind_m[0][0]) & (ind_d[1][0] < ind_m[1][0]):
                    direcs.append(2)
                elif (ind_d[0][0] < ind_m[0][0]) & (ind_d[1][0] == ind_m[1][0]):
                    direcs.append(3)
                elif (ind_d[0][0] > ind_m[0][0]) & (ind_d[1][0] == ind_m[1][0]):
                    direcs.append(4)
                elif (ind_d[0][0] < ind_m[0][0]) & (ind_d[1][0] > ind_m[1][0]):
                    direcs.append(5)
                elif (ind_d[0][0] >= ind_m[0][0]) & (ind_d[1][0] > ind_m[1][0]):
                    direcs.append(6)
            else: 
                if (ind_d[0][0] <= ind_m[0][0]) & (ind_d[1][0] < ind_m[1][0]):
                    direcs.append(1)
                elif (ind_d[0][0] > ind_m[0][0]) & (ind_d[1][0] < ind_m[1][0]):
                    direcs.append(2)
                elif (ind_d[0][0] < ind_m[0][0]) & (ind_d[1][0] == ind_m[1][0]):
                    direcs.append(3)
                elif (ind_d[0][0] > ind_m[0][0]) & (ind_d[1][0] == ind_m[1][0]):
                    direcs.append(4)
                elif (ind_d[0][0] <= ind_m[0][0]) & (ind_d[1][0] > ind_m[1][0]):
                    direcs.append(5)
                elif (ind_d[0][0] > ind_m[0][0]) & (ind_d[1][0] > ind_m[1][0]):
                    direcs.append(6)
        #for i in range(3):
        #    print(moves[i], nums[i], direcs[i])
        if str([moves, nums, direcs]) in agent.root.children.keys():
            mapstate = deepcopy(agent.root_mapstate)
            sheepstate = deepcopy(agent.root_sheepstate)
            for i in range(3):
                if direcs[i] != -1:
                    sheepstate, mapstate = play((player + i) % 4 + 1, mapstate, sheepstate, [moves[i], nums[i], direcs[i]])
            agent.root = agent.root.children[str([moves, nums, direcs])]
            agent.root_mapstate = mapStat
            agent.root_sheepstate = sheepStat
        else:
            agent.set_gamestate(mapStat, sheepStat)
    agent.search(search_time)
    #num_rollouts, node_count, run_time = agent.statistics()
    move = agent.best_move()  # the move is tuple like (3, 1)
    step = [(move[0][0], move[0][1]), move[1], move[2]]
    agent.move(move)
    #print("time: ", clock() - start)
    return step


# player initial
(id_package, playerID, mapStat) = STcpClient.GetMap()
init_pos = InitPos(mapStat)
STcpClient.SendInitPos(id_package, init_pos)

# start game
while (True):
    (end_program, id_package, mapStat, sheepStat) = STcpClient.GetBoard()
    if end_program:
        STcpClient._StopConnect()
        break
    Step = GetStep(playerID, mapStat, sheepStat)

    STcpClient.SendStep(id_package, Step)
