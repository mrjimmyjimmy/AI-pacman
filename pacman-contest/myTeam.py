# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# # educational purposes provided that (1) you do not distribute or publish
# # solutions, (2) you retain this notice, and (3) you provide clear
# # attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# #
# # Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# # The core projects and autograders were primarily created by John DeNero
# # (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# # Student side autograding was added by Brad Miller, Nick Hay, and
# # Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game


from math import sqrt, log
import random, time
import baselineTeam

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='DummyAgent', second='DummyAgent'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

class DummyAgent(CaptureAgent):
    """
    A Dummy agent to serve as an example of the necessary agent structure.
    You should look at baselineTeam.py for more details about how to
    create an agent as this is the bare minimum.
    """

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).

        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)

        IMPORTANT: This method may run for at most 15 seconds.
        """

        '''
        Make sure you do not delete the following line. If you would like to
        use Manhattan distances instead of maze distances in order to save
        on initialization time, please take a look at
        CaptureAgent.registerInitialState in captureAgents.py.
        '''
        CaptureAgent.registerInitialState(self, gameState)

        '''
        Your initialization code goes here, if you need any.
        '''

    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """
        actions = gameState.getLegalActions(self.index)

        '''
        You should change this in your own agent.
        '''

        return random.choice(actions)

    def testGitLab(self):
        return 100


class MCTree:
    def __init__(self, gameState, ancestor=None, agentindex=0):
        self.gameState = gameState
        self.subtrees = {}
        # self.subtreesCounter = {}
        self.ancestor = ancestor
        self.depth = 0
        self.visited = 0
        self.score = 0
        self.agentindex = agentindex

        if ancestor is not None:
            self.depth = ancestor.depth + 1


    def __str__(self):
        print "Depth:", self.depth, "\nSubtrees:", len(self.subtrees), \
            "\nVisited:", str(self.visited), "\nScore:", str(self.score)
        return ""


    def expand(self):
        # print "Expanding..."
        if len(self.subtrees) == 0:
            for action in self.gameState.getLegalActions(self.agentindex):
                self.subtrees[action] = MCTree(self.gameState.generateSuccessor(self.agentindex, action), self, self.agentindex)
            print "expanded tree:", self
                # self.subtreesCounter[action] = 0
            # print "Expanded"
        # else:
            # print "No expand"
        # else:


    def tree_policy(self):
        actions = self.gameState.getLegalActions(self.agentindex)
        if len(actions) == 1:
            return actions[0]

        maxQ = -10000
        maxQaction = Directions.STOP
        CP = 1/2
        for action in actions:
            # if self.subtreesCounter[action] == 0:
            #     self.subtreesCounter[action] += 1
            if self.subtrees[action].visited == 0:
                print "Tree policy returning new:", action
                return action
            uct = evl(self.gameState.generateSuccessor(self.agentindex, action)) + 2*CP* sqrt(2*log(self.visited)/self.subtrees[action].visited)
            if uct >= maxQ:
                maxQ = uct
                maxQaction = action
        # self.subtreesCounter[action] += 1
        print "Tree policy returning: ", maxQaction
        return maxQaction


    def backprop(self, simulate_score):
        self.visited += 1
        self.score += simulate_score
        if self.ancestor is not None:
            print "from depth", self.depth
            print "backprop to ancestor: ", simulate_score
            self.ancestor.backprop(simulate_score)
        else:
            print "Backproped to root"
            print "the depth is:", self.depth
            print "root socre:", self.score



def random_simulation(gameState, agentindex, depth):
        if depth > 0:
            actions = gameState.getLegalActions(agentindex)
            reverse_direction = Directions.REVERSE[gameState.getAgentState(agentindex).getDirection()]
            if len(actions) > 1:
                # print "Actions before remove: ", actions
                # for action in actions:
                #     print type(action)
                # print "Reverse direction: ", reverse_direction, type(reverse_direction)
                actions.remove(reverse_direction)
                # print "Actions: ", actions
                action = random.choice(actions)
                # print "Action taken:", action
            else:
                action = actions[0]
            random_simulation(gameState.generateSuccessor(agentindex, action), agentindex, depth - 1)
        return evl(gameState)


def MCTsearch(gameState, agentindex, depth):
    start_time = time.time()
    print "Starting Search time: ", start_time
    root = MCTree(gameState, None, agentindex)
    root.expand()

    while time.time() - start_time < 0.08:  # budget time
        print "New simulation! Time Elapsed: ", time.time() - start_time
        action = root.tree_policy()
        tree = root.subtrees[action]
        print "number of taking the action on root:", tree.visited
        while not tree.visited == 0:
            print "subtree depth:", tree.depth
            # print tree
            # print "Calling for expansion"
            tree.expand()   # effective if len(subtrees) == 0
            # print "expanded tree: ", tree
            action = tree.tree_policy()
            tree = tree.subtrees[action]
        simulated_score = random_simulation(tree.gameState, agentindex, depth)
        print "simulated score:", simulated_score
        tree.backprop(simulated_score)
        print "After backprop, root:", root

    maxQ = -10000
    maxQaction = None
    for action in root.gameState.getLegalActions(root.agentindex):
        subtree = root.subtrees[action]
        if subtree.visited == 0:
            print "Error, an action is never simulated:", action
            continue
        qvalue = subtree.score / subtree.visited
        if qvalue >= maxQ:
            maxQ = qvalue
            maxQaction = action
    print "********RETURN ACTION*********", maxQaction
    print root
    # exit(100)
    return maxQaction



def evl(gameState):
    value = random.uniform(0,1)
    # print "random value:", value
    return value



class myoffensiveagent(baselineTeam.ReflexCaptureAgent):
    def chooseAction(self, gameState):
        return MCTsearch(gameState, self.index, 5)

def createTeam(firstIndex, secondIndex, isRed,
               first = 'myoffensiveagent', second = 'myoffensiveagent'):

  return [eval(first)(firstIndex), eval(second)(secondIndex)]