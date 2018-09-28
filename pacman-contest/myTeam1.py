from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util
from game import Directions
import game
from util import nearestPoint



from math import sqrt, log
import random, time
import baselineTeam

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent'):
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
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)

        self.distancer.getMazeDistances()
        if self.red:
            centralX = (gameState.data.layout.width - 2) / 2
        else:
            centralX = ((gameState.data.layout.width - 2) / 2) + 1
        self.boundary = []
        for i in range(1, gameState.data.layout.height - 1):
            if not gameState.hasWall(centralX, i):
                self.boundary.append((centralX, i))
        # self.weights = {'score': 0, 'DisToNearestFood': 0, 'disToGhost': 0, 'disToCapsule': 0, 'dots': 0,
        #            'disToBoundary': -5}


    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights




class OffensiveReflexAgent(ReflexCaptureAgent):

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)

        self.distancer.getMazeDistances()
        if self.red:
            centralX = (gameState.data.layout.width - 2) / 2
        else:
            centralX = ((gameState.data.layout.width - 2) / 2) + 1
        self.boundary = []
        for i in range(1, gameState.data.layout.height - 1):
            if not gameState.hasWall(centralX, i):
                self.boundary.append((centralX, i))
        self.weights = {'score': 0, 'DisToNearestFood': 0, 'disToGhost': 0, 'disToCapsule': 0, 'stake':0}
        # self.weights = {'score': 0, 'DisToNearestFood': -5, 'disToGhost': 50, 'disToCapsule': -55, 'dots': 50,
        #            'disToBoundary': -50}

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        epislon = 0   # the chanse to randomly choose an action - going to 0 at last

        # return MCTsearch(gameState, self, depth=5)

        actions = gameState.getLegalActions(self.index)
        actions.remove(Directions.STOP)

        if util.flipCoin(epislon):
            action = random.choice(actions)
            self.updateWeights(gameState,action)
            return action

        maxQ = -float("inf")
        maxQaction = None
        for action in actions:
            qval = self.evl(self.getSuccessor(gameState,action))
            if qval >= maxQ:
                maxQ = qval
                maxQaction = action
        self.updateWeights(gameState, maxQaction)
        return maxQaction

    def getReward(self, gameState, action):
        reward = 0
        nextState = self.getSuccessor(gameState, action)
        score = nextState.getScore() - gameState.getScore()
        stepCost = -0.5
        foodReward = 0.8
        disToGhost = self.disToNearestGhost(gameState)
        food = self.getFood(gameState)
        dx, dy = nextState.getAgentState(self.index).getPosition()

        if food[int(dx)][int(dy)]:
            reward += foodReward

        if disToGhost <= 1:
            reward += -5
        return reward + stepCost + score

    def disToNearestGhost(self, gameState):
        agentPosition = gameState.getAgentState(self.index).getPosition()
        enemies = []
        for e in self.getOpponents(gameState):
            enemyState = gameState.getAgentState(e)
            if not enemyState.isPacman and not enemyState.getPosition() is None:
                enemies.append(enemyState)

        if len(enemies) > 0:
            toEnemies = []
            for e in enemies:
                enemyPos = e.getPosition()
                toEnemies.append(self.getMazeDistance(agentPosition, enemyPos))
            # closest = min(position, key=lambda x: self.agent.getMazeDistance(agentPosition, x))

            dis = min(toEnemies)
            if dis < 6:
                return dis
        else:
            dis = []
            # dis = (dis.append(gameState.getAgentDistances()[index]) for index in self.agent.getOpponents(gameState))
            for index in self.getOpponents(gameState):
                dis.append(gameState.getAgentDistances()[index])
            return min(dis)


    def evl(self, gameState):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState)
        weights = self.getWeights(gameState)
        return features * weights

    def getFeatures(self, gameState):
        features = util.Counter()
        agentPosition = gameState.getAgentState(self.index).getPosition()

        # ---------------------feature 1: score----------------
        features['score'] = self.getScore(gameState)
        # ---------------------feature 2: distance to closest food----------------
        food = self.getFood(gameState).asList()
        if len(food) > 0:
            dis = []
            for f in food:
                dis.append(self.getMazeDistance(agentPosition, f))
            minDis = min(dis)
            features['DisToNearestFood'] = minDis

        # ---------------------feature 3: dis to closest ghost----------------
        enemies = []
        for e in self.getOpponents(gameState):
            enemyState = gameState.getAgentState(e)
            if not enemyState.isPacman and not enemyState.getPosition() is None:
                enemies.append(enemyState)

        if len(enemies) > 0:
            toEnemies = []
            for e in enemies:
                enemyPos = e.getPosition()
                toEnemies.append(self.getMazeDistance(agentPosition, enemyPos))
            # closest = min(position, key=lambda x: self.agent.getMazeDistance(agentPosition, x))

            dis = min(toEnemies)
            if dis < 6:
                features['disToGhost'] = dis
        else:
            dis = []
            # dis = (dis.append(gameState.getAgentDistances()[index]) for index in self.agent.getOpponents(gameState))
            for index in self.getOpponents(gameState):
                dis.append(gameState.getAgentDistances()[index])
            features['disToGhost'] = min(dis)

        # ---------------------feature 4: dis to closest capsule----------------
        capsule = self.getCapsules(gameState)
        if len(capsule) == 0:
            features['disToCapsule'] = 0
        else:
            dis = []
            for c in capsule:
                dis.append(self.getMazeDistance(agentPosition, c))
            features['disToCapsule'] = min(dis)
        # ---------------------feature 5: carrying----------------
        # features['dots'] = gameState.getAgentState(self.index).numCarrying

        dots = gameState.getAgentState(self.index).numCarrying
        # ---------------------feature 6: dis to boundary----------------
        disToBoundary = 99999
        for a in range(len(self.boundary)):
            disToBoundary = min(disToBoundary, self.getMazeDistance(agentPosition, self.boundary[a]))
        # features['disToBoundary'] = disToBoundary

        features['stake'] = disToBoundary*dots

        # ---------------------feature 7: dis to opponent's attackers----------------
        #  need more work on this feature

        return features

    def getWeights(self, gameState):
        # weights = {'score': 0, 'DisToNearestFood': 0, 'disToGhost': 0, 'disToCapsule': 0, 'dots': 0,
        #            'disToBoundary': 0}
        #
        # enemies = []
        # for e in self.getOpponents(gameState):
        #     enemyState = gameState.getAgentState(e)
        #     if not enemyState.isPacman and not enemyState.getPosition() is None:
        #         enemies.append(enemyState)
        return self.weights

    def getMaxQ(self, gameState):
        Qvalue = []
        actions = gameState.getLegalActions(self.index)
        for a in actions:
            nextState = self.getSuccessor(gameState,a)
            Qvalue.append(self.evl(nextState))
        return max(Qvalue)

    def updateWeights(self, gameState, action):
        alpha = 0.001
        discount = 0.8
        nextState = self.getSuccessor(gameState,action)
        features = self.getFeatures(nextState)

        reward = self.getReward(gameState,action)
        maxQ = self.getMaxQ(nextState)
        q = self.evl(gameState)


        for f in features:


            self.weights[f] += alpha * (reward + discount * maxQ - q) * features[f]



#########################
##  Muiltitype agents  ##
#########################

class DefensiveReflexAgent(ReflexCaptureAgent):


    def atCenter(self, myPos):
        minDis = []
        for boundary in self.boundary:
            minDis.append(self.getMazeDistance(myPos, boundary))
            minDis.sort()
        if minDis[0] > 2:
            return False
        else:
            return True

    # return a list of (x,y), shows the enemies positions
    def getEnemy(self, gameState):
        enemyPos = []
        enemies = self.getOpponents(gameState)
        for enemy in enemies:
            pos = gameState.getAgentPosition(enemy)
            if pos != None:
                enemyPos.append((enemy,pos))

        return enemyPos


    # Find which enemy is the closest
    def getEnemyDis(self, gameState, myPos):
        pos = self.getEnemy(gameState)
        minDist = None
        if len(pos) > 0:
            minDist = 6
            # myPos = gameState.getAgentPosition(self.index)
            for i, p in pos:
                dist = self.getMazeDistance(p, myPos)
                if dist < minDist:
                    minDist = dist
        return minDist

    # How much longer is the ghost scared?
    def ScaredTimer(self, gameState):
        return gameState.getAgentState(self.index).scaredTimer

    # return a list of (x,y) shows the food lost in our side
    def foodLostPosition(self):
        currentState = self.getCurrentObservation()
        previousState = self.getPreviousObservation()
        currentFood = self.getFoodYouAreDefending(currentState)
        previousFood = None
        if previousState != None:
            previousFood = self.getFoodYouAreDefending(previousState)
            if not currentFood == previousFood:
                food = self.differInMatrax(currentFood, previousFood)
                return food

    # gice the difference element of two matrax
    def differInMatrax(self, current, previous):

        list_curr = []
        list_pre = []
        foodPos = []
        for i in current:
            list_curr.append(i)
        for i in previous:
            list_pre.append(i)

        for i in range (len(list_pre)):
            if not list_pre[i] == list_curr[i]:
                for j in range(len(list_curr[i])):
                    if not list_curr[i][j] == list_pre[i][j]:
                        foodPos.append((i,j))

        return foodPos

    def evaluate(self, gameState, action, agentType):

        """
        Computes a linear combination of features and feature weights
        """
        return {
            'inital': self.getFeatureInital(gameState,action) * self.getWeightsInital(gameState,action),
            'defence': self.getFeaturesDefence(gameState,action) * self.getWeightsDefence(gameState,action),
            'offence': self.getFeatureOffence(gameState,action) * self.getWeightOffence(gameState, action),
        }.get(agentType)

    def getFeatureOffence(self, gameState, action):

        # initial features
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        # get the position
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()


        # ---------------------feature 1: score----------------
        features['score'] = self.getScore(gameState)
        # ---------------------feature 2: distance to closest food----------------
        food = self.getFood(gameState).asList()
        if len(food) > 0:
            dis = []
            for f in food:
                dis.append(self.getMazeDistance(myPos, f))
            minDis = min(dis)
            features['DisToNearestFood'] = minDis

        # ---------------------feature 3: dis to closest ghost----------------
        enemies = []
        for e in self.getOpponents(gameState):
            enemyState = gameState.getAgentState(e)
            if not enemyState.isPacman and not enemyState.getPosition() is None:
                enemies.append(enemyState)

        if len(enemies) > 0:
            toEnemies = []
            for e in enemies:
                enemyPos = e.getPosition()
                toEnemies.append(self.getMazeDistance(myPos, enemyPos))
            # closest = min(position, key=lambda x: self.agent.getMazeDistance(agentPosition, x))

            dis = min(toEnemies)
            if dis < 6:
                features['disToGhost'] = dis
        else:
            dis = []
            # dis = (dis.append(gameState.getAgentDistances()[index]) for index in self.agent.getOpponents(gameState))
            for index in self.getOpponents(gameState):
                dis.append(gameState.getAgentDistances()[index])
            features['disToGhost'] = min(dis)

        # ---------------------feature 4: dis to closest capsule----------------
        capsule = self.getCapsules(gameState)
        if len(capsule) == 0:
            features['disToCapsule'] = 0
        else:
            dis = []
            for c in capsule:
                dis.append(self.getMazeDistance(myPos, c))
            features['disToCapsule'] = min(dis)
        # ---------------------feature 5: carrying----------------
        features['dots'] = gameState.getAgentState(self.index).numCarrying
        # ---------------------feature 6: dis to boundary----------------
        disToBoundary = 99999
        for a in range(len(self.boundary)):
            disToBoundary = min(disToBoundary, self.getMazeDistance(myPos, self.boundary[a]))
        features['disToBoundary'] = disToBoundary
        # ---------------------feature 7: dis to opponent's attackers----------------
        #  need more work on this feature

        return features

    def getWeightOffence(self, gameState, action):

        return {'score': 30, 'DisToNearestFood': -5, 'disToGhost': 50, 'disToCapsule': -55, 'dots': 50,
                'disToBoundary': -50}

    def getFeatureInital(self, gameState, action):

        # inital features
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        # get the position
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()
        disToBoundary = 99999
        for a in range(len(self.boundary)):
            disToBoundary = min(disToBoundary, self.getMazeDistance(myPos, self.boundary[a]))
        features['distToBoundary'] = disToBoundary

        return features

    def getWeightsInital(self, gameState, action):

        return {'disToBoundary' : 100}

    def getFeaturesDefence(self, gameState, action):

        # initial features
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        # get the position
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]

        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        # get the distance to enemy
        enemyDistance = self.getEnemyDis(successor, myPos)
        if(enemyDistance <= 5):
            features['danger'] = 1
            if(enemyDistance <= 1 and self.ScaredTimer(successor) > 0):
                features['danger'] = -1
        else:
            features['danger'] = 0

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        # get the distance where the food lost
        if not self.foodLostPosition()  == None:
            food = self.foodLostPosition()
            if 0 < len(food) < 3:
                food = food[0]
                minDistance = self.getMazeDistance(myPos, food)
                self.foodLost = minDistance
                features['lostFoodDistance'] = minDistance

        disToBoundary = 99999
        for a in range(len(self.boundary)):
            disToBoundary = min(disToBoundary, self.getMazeDistance(myPos, self.boundary[a]))
        features['distToBoundary'] = disToBoundary

        return features

    def getWeightsDefence(self, gameState, action):
        return {'numInvaders': -1000, 'onDefense': 1, 'invaderDistance': -10000, 'stop': -1, 'reverse': -1, 'danger': 1
            , 'lostFoodDistance': -100, 'distToBoundary': -1}



    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        # get all legal actions
        actions = gameState.getLegalActions(self.index)
        myPos = gameState.getAgentPosition(self.index)
        opponents = self.getOpponents(gameState)

        """
        switch agent type here
        """
        agentType = 'offence'

        # there is no enemy, try to catch some food
        for enemy in opponents:
            if gameState.getAgentState(enemy).isPacman:
                agentType = 'defence'

        values = [self.evaluate(gameState, a, agentType) for a in actions]
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        return random.choice(bestActions)


##########
# Agents #
##########

class MCTree:
    def __init__(self, gameState, ancestor=None, agent=None):
        self.gameState = gameState
        self.subtrees = {}
        # self.subtreesCounter = {}
        self.ancestor = ancestor
        self.depth = 0
        self.visited = 0
        self.score = 0
        self.agent = agent
        self.agentindex = agent.index

        if ancestor is not None:
            self.depth = ancestor.depth + 1




    def expand(self):
        # print "Expanding..."
        if len(self.subtrees) == 0:
            for action in self.gameState.getLegalActions(self.agentindex):
                self.subtrees[action] = MCTree(self.gameState.generateSuccessor(self.agentindex, action), self, self.agent)
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
                return action
            uct = self.agent.evl(self.gameState.generateSuccessor(self.agentindex, action)) + 2*CP* sqrt(2*log(self.visited)/self.subtrees[action].visited)
            if uct >= maxQ:
                maxQ = uct
                maxQaction = action
        # self.subtreesCounter[action] += 1
        return maxQaction


    def backprop(self, simulate_score):
        self.visited += 1
        self.score += simulate_score
        if self.ancestor is not None:
            # print "from depth", self.depth
            # print "backprop to ancestor: ", simulate_score
            self.ancestor.backprop(simulate_score)



def random_simulation(gameState, agent, depth):
    # print "agent in simulation", agent
    agentindex = agent.index
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
        random_simulation(gameState.generateSuccessor(agentindex, action), agent, depth - 1)
    return agent.evl(gameState)


def MCTsearch(gameState, agent, depth):
    start_time = time.time()
    # print "Starting Search time: ", start_time
    root = MCTree(gameState, None, agent)
    root.expand()

    while time.time() - start_time < 0.08:  # budget time
        # print "New simulation! Time Elapsed: ", time.time() - start_time
        action = root.tree_policy()
        tree = root.subtrees[action]
        # print "number of taking the action on root:", tree.visited
        while not tree.visited == 0:
            # print "subtree depth:", tree.depth
            # print tree
            # print "Calling for expansion"
            tree.expand()   # effective if len(subtrees) == 0
            # print "expanded tree: ", tree
            action = tree.tree_policy()
            tree = tree.subtrees[action]
        simulated_score = random_simulation(tree.gameState, agent, depth)
        # print "simulated score:", simulated_score
        tree.backprop(simulated_score)
        # print "After backprop, root:", root

    maxQ = -10000
    maxQaction = None
    for action in root.gameState.getLegalActions(root.agentindex):
        subtree = root.subtrees[action]
        if subtree.visited == 0:
            # print "Error, an action is never simulated:", action
            continue
        qvalue = subtree.score / subtree.visited
        if qvalue >= maxQ:
            maxQ = qvalue
            maxQaction = action
    # exit(100)
    return maxQaction


#
# def evl(gameState):
#     value = random.uniform(0,1)
#     # print "random value:", value
#     return value


#
# class myoffensiveagent(baselineTeam.ReflexCaptureAgent):
#     def chooseAction(self, gameState):
#         return MCTsearch(gameState, self.index, 5)

