from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util
from game import Directions
import game
from util import nearestPoint
import copy
from math import sqrt, log
import random, time
import baselineTeam


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveReflexAgent', second='DefensiveReflexAgent'):
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

    def legalPosition(self, previousPosition, nextPosition):
        x, y = previousPosition
        m, n = nextPosition
        if (m == x + 1 or m == x - 1) and (n == y + 1 or n == y - 1):
            return True
        else:
            return False

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)

        self.weights = {'score': 1.78261354182, 'DisToNearestFood': -4.91094492098, 'disToGhost': 8.17572535548,
                        'disToCapsule': -1.36111562824, 'dots': -0.877933155097,
                        'disToBoundary': -2.94156916302, 'deadends': -10}

        # self.weights = {'score': 0, 'DisToNearestFood': 0, 'disToGhost':0,
        #                 'disToCapsule': 0, 'dots': 0,
        #                 'disToBoundary': 0,'deadends':0}
        self.distancer.getMazeDistances()

        # ----------- DEADEND PROCESSING
        self.deadEnds = {}

        neighbors = {}
        walkable = []
        for i in range(1, gameState.data.layout.width - 1):
            for j in range(1, gameState.data.layout.height - 1):
                if not gameState.hasWall(i, j):
                    walkable.append((i, j))

        for (i, j) in walkable:
            print "walkable ij", (i, j)
            neighbor = []
            if (i + 1, j) in walkable:
                neighbor.append(Directions.EAST)
                print "Adding neighbour:", (i + 1, j)
            if (i - 1, j) in walkable:
                neighbor.append(Directions.WEST)
                print "Adding neighbour:", (i - 1, j)
            if (i, j + 1) in walkable:
                neighbor.append(Directions.NORTH)
                print "Adding neighbour:", (i, j + 1)
            if (i, j - 1) in walkable:
                neighbor.append(Directions.SOUTH)
                print "Adding neighbour:", (i, j - 1)

            neighbors[(i, j)] = neighbor

        for (i, j) in neighbors:
            if len(neighbors[(i, j)]) >= 3:
                print "ij:", (i, j)
                print "neighbours:", neighbors[(i, j)]
                for direction in neighbors[(i, j)]:
                    (i1, j1), revdir = nextStep((i, j), direction)
                    nextNeighbor = neighbors[(i1, j1)]
                    if len(nextNeighbor) >= 3:
                        continue
                    elif len(nextNeighbor) == 1:
                        self.deadEnds[(i, j), direction] = 1


                    else:
                        depth = 1
                        while len(nextNeighbor) == 2:
                            depth += 1
                            nextNeighbor.remove(revdir)
                            nextNeighbor.append(revdir)
                            (i1, j1), revdir = nextStep((i1, j1), nextNeighbor[0])
                            nextNeighbor = neighbors[(i1, j1)]
                            if len(nextNeighbor) >= 3:
                                continue
                            elif len(nextNeighbor) == 1:
                                self.deadEnds[(i, j), direction] = depth

        for deadend in self.deadEnds:
            print "Correct deadends: ", deadend, self.deadEnds[deadend]

        # -----------

        print "Map width:", gameState.data.layout.width
        if self.red:
            cX = (gameState.data.layout.width - 2) / 2
            self.deadEnds = dict((((x, y), dir), self.deadEnds[((x, y), dir)]) for ((x, y), dir) in self.deadEnds if
                                 x > cX)  # deadend filter
            print "blue deadends:", self.deadEnds
        else:
            cX = ((gameState.data.layout.width - 2) / 2) + 1
            self.deadEnds = dict((((x, y), dir), self.deadEnds[((x, y), dir)]) for ((x, y), dir) in self.deadEnds if
                                 x < cX)  # deadend filter
            print "red deadends:", self.deadEnds

        self.boundary = []
        for i in range(1, gameState.data.layout.height - 1):
            if not gameState.hasWall(cX, i):
                self.boundary.append((cX, i))

        # self.weights = {'score': 0, 'DisToNearestFood': -5, 'disToGhost': 50, 'disToCapsule': -55, 'dots': 50,
        #            'disToBoundary': -50}

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

    def getFeaturesOffense(self, previous, action):
        features = util.Counter()
        gameState = self.getSuccessor(previous, action)
        previousPosition = previous.getAgentState(self.index).getPosition()
        agentPosition = gameState.getAgentState(self.index).getPosition()
        if self.legalPosition(previousPosition, agentPosition):
            features['disToGhost'] = -100
            return features

        # ---------------------feature 1: score----------------
        features['score'] = self.getScore(gameState)
        # ---------------------feature 2: distance to closest food----------------
        x, y = gameState.getAgentState(self.index).getPosition()
        oldFood = self.getFood(previous)
        if oldFood[int(x)][int(y)]:
            features['DisToNearestFood'] = 0
        else:
            food = self.getFood(gameState).asList()
            if len(food) <= 2:
                features['DisToNearestFood'] = 0
            if len(food) > 2:
                dis = []
                for f in food:
                    dis.append(self.getMazeDistance(agentPosition, f))
                minDis = min(dis)
                features['DisToNearestFood'] = minDis

        # ---------------------feature 3: dis to closest ghost----------------
        enemies = []
        for e in self.getOpponents(gameState):
            enemyState = gameState.getAgentState(e)
            if not enemyState.isPacman and not enemyState.getPosition() is None and not enemyState.scaredTimer > 5:
                enemies.append(enemyState)

        if len(enemies) > 0:
            print "FFFFFFFFFFFFFFFFFFFF Length of enemies > 0"
            toEnemies = []
            for e in enemies:
                enemyPos = e.getPosition()
                toEnemies.append(self.getMazeDistance(agentPosition, enemyPos))
            # closest = min(position, key=lambda x: self.agent.getMazeDistance(agentPosition, x))

            dis = min(toEnemies)
            features['disToGhost'] = dis
            if dis >= 12:
                features['disToGhost'] = 12
        else:
            features['disToGhost'] = 12
            # dis = []
            # # dis = (dis.append(gameState.getAgentDistances()[index]) for index in self.agent.getOpponents(gameState))
            # for index in self.getOpponents(gameState):
            #     dis.append(gameState.getAgentDistances()[index])
            # print "FFFFFFFFFFFFFFFFFFF length of enemies < 0, dis[]:", dis
            # if min(dis) <= 6 :
            #     features['disToGhost'] = 6
            # else:
            #     features['disToGhost'] = min(dis)

        # features['disToGhost'] = 1/features['disToGhost']

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
        features['dots'] = gameState.getAgentState(self.index).numCarrying
        # ---------------------feature 6: dis to boundary----------------
        disToBoundary = 99999
        for a in range(len(self.boundary)):
            disToBoundary = min(disToBoundary, self.getMazeDistance(agentPosition, self.boundary[a]))
        features['disToBoundary'] = disToBoundary
        # ---------------------feature 7: dis to opponent's attackers----------------
        #  need more work on this feature

        # ---------------------feature 7: remaining food----------------

        # ----------------------feature 8: deadends-----------
        features['deadends'] = 0
        if self.deadEnds.has_key((previous.getAgentState(self.index).getPosition(), action)) and self.deadEnds[
            (previous.getAgentState(self.index).getPosition(), action)] * 2 >= features['disToGhost'] - 1 > 0:
            features['deadends'] = 100
        # features.divideAll(10)

        return features


def nextStep((x, y), direction):
    if direction == Directions.SOUTH:
        return ((x, y - 1), Directions.NORTH)
    elif direction == Directions.NORTH:
        return ((x, y + 1), Directions.SOUTH)
    elif direction == Directions.EAST:
        return ((x + 1, y), Directions.WEST)
    else:
        return ((x - 1, y), Directions.EAST)


class OffensiveReflexAgent(ReflexCaptureAgent):

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)

        self.weights = {'score': 1.78261354182, 'DisToNearestFood': -4.91094492098, 'disToGhost': 8.17572535548,
                        'disToCapsule': -1.36111562824, 'dots': -0.877933155097,
                        'disToBoundary': -2.94156916302, 'deadends': -10}

        # self.weights = {'score': 0, 'DisToNearestFood': 0, 'disToGhost':0,
        #                 'disToCapsule': 0, 'dots': 0,
        #                 'disToBoundary': 0,'deadends':0}
        self.distancer.getMazeDistances()

        # ----------- DEADEND PROCESSING
        self.deadEnds = {}

        neighbors = {}
        walkable = []
        for i in range(1, gameState.data.layout.width - 1):
            for j in range(1, gameState.data.layout.height - 1):
                if not gameState.hasWall(i, j):
                    walkable.append((i, j))

        for (i, j) in walkable:
            print "walkable ij", (i, j)
            neighbor = []
            if (i + 1, j) in walkable:
                neighbor.append(Directions.EAST)
                print "Adding neighbour:", (i + 1, j)
            if (i - 1, j) in walkable:
                neighbor.append(Directions.WEST)
                print "Adding neighbour:", (i - 1, j)
            if (i, j + 1) in walkable:
                neighbor.append(Directions.NORTH)
                print "Adding neighbour:", (i, j + 1)
            if (i, j - 1) in walkable:
                neighbor.append(Directions.SOUTH)
                print "Adding neighbour:", (i, j - 1)

            neighbors[(i, j)] = neighbor

        for (i, j) in neighbors:
            if len(neighbors[(i, j)]) >= 3:
                print "ij:", (i, j)
                print "neighbours:", neighbors[(i, j)]
                for direction in neighbors[(i, j)]:
                    (i1, j1), revdir = nextStep((i, j), direction)
                    nextNeighbor = neighbors[(i1, j1)]
                    if len(nextNeighbor) >= 3:
                        continue
                    elif len(nextNeighbor) == 1:
                        self.deadEnds[(i, j), direction] = 1


                    else:
                        depth = 1
                        while len(nextNeighbor) == 2:
                            depth += 1
                            nextNeighbor.remove(revdir)
                            nextNeighbor.append(revdir)
                            (i1, j1), revdir = nextStep((i1, j1), nextNeighbor[0])
                            nextNeighbor = neighbors[(i1, j1)]
                            if len(nextNeighbor) >= 3:
                                continue
                            elif len(nextNeighbor) == 1:
                                self.deadEnds[(i, j), direction] = depth

        for deadend in self.deadEnds:
            print "Correct deadends: ", deadend, self.deadEnds[deadend]

        # -----------

        print "Map width:", gameState.data.layout.width
        if self.red:
            cX = (gameState.data.layout.width - 2) / 2
            self.deadEnds = dict((((x, y), dir), self.deadEnds[((x, y), dir)]) for ((x, y), dir) in self.deadEnds if
                                 x > cX)  # deadend filter
            print "blue deadends:", self.deadEnds
        else:
            cX = ((gameState.data.layout.width - 2) / 2) + 1
            self.deadEnds = dict((((x, y), dir), self.deadEnds[((x, y), dir)]) for ((x, y), dir) in self.deadEnds if
                                 x < cX)  # deadend filter
            print "red deadends:", self.deadEnds

        self.boundary = []
        for i in range(1, gameState.data.layout.height - 1):
            if not gameState.hasWall(cX, i):
                self.boundary.append((cX, i))

        # self.weights = {'score': 0, 'DisToNearestFood': -5, 'disToGhost': 50, 'disToCapsule': -55, 'dots': 50,
        #            'disToBoundary': -50}

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        epislon = 0  # the chanse to randomly choose an action - going to 0 at last

        print "agent:", self
        print "agent index", self.index
        # return MCTsearch(gameState, self, depth=5)

        actions = gameState.getLegalActions(self.index)
        actions.remove(Directions.STOP)

        if util.flipCoin(epislon):
            action = random.choice(actions)
            # self.updateWeights(gameState, action)
            return action

        maxQ = -float("inf")
        maxQaction = None
        for action in actions:
            qval = self.evl2(gameState, action)
            print "action", action
            print qval
            if qval >= maxQ:
                maxQ = qval
                maxQaction = action

        # self.updateWeights(gameState, maxQaction)
        print "====================================]=================so i choose:", maxQaction
        return maxQaction

    def getReward(self, gameState, action):
        reward = 0
        nextState = self.getSuccessor(gameState, action)
        score = nextState.getScore() - gameState.getScore()
        stepCost = -0.2
        foodReward = 1
        disToGhost = self.disToNearestGhost(gameState)
        food = self.getFood(gameState)
        dx, dy = nextState.getAgentState(self.index).getPosition()
        dots = gameState.getAgentState(self.index).numCarrying

        if food[int(dx)][int(dy)]:
            reward += foodReward

        capsule = self.getCapsules(nextState)
        if len(capsule) and capsule[0] == (dx, dy):
            reward += 10

        if disToGhost <= 2:
            reward += -20 + dots * -2

        agentPosition = gameState.getAgentState(self.index).getPosition()
        if self.deadEnds.has_key((agentPosition, action)) and self.deadEnds[(agentPosition, action)] * 2 > disToGhost:
            reward -= 200

        disToBoundary = 99999
        for a in range(len(self.boundary)):
            disToBoundary = min(disToBoundary, self.getMazeDistance(agentPosition, self.boundary[a]))

        return reward + stepCost + 20 * score - 1 * disToBoundary * dots

    def disToNearestGhost(self, gameState):
        agentPosition = gameState.getAgentState(self.index).getPosition()
        enemies = []
        for e in self.getOpponents(gameState):
            enemyState = gameState.getAgentState(e)
            if not enemyState.isPacman and not enemyState.getPosition() is None and not enemyState.scaredTimer > 5:
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
            return 6

    # def evl(self, gameState):
    #     """
    #     Computes a linear combination of features and feature weights
    #     """
    #     features = self.getFeatures(gameState)
    #     weights = self.getWeights(gameState)
    #     return features * weights

    def evl2(self, gameState, action):
        features = self.getFeaturesOffense(gameState, action)
        weights = self.getWeights(gameState, action)
        newWeights = copy.deepcopy(weights)

        # if features['dots'] == 0:
        #     newWeights['disToBoundary'] *= 0.7
        # else:
        #     newWeights['disToBoundary'] *= (1.0 + float(features['dots']) / 83)

        print features
        print newWeights
        return features * newWeights



    def getWeights(self, gameState, action):
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
            Qvalue.append(self.evl2(gameState, a))
        return max(Qvalue)

    def updateWeights(self, gameState, action):
        alpha = 0.0001
        discount = 0.7
        nextState = self.getSuccessor(gameState, action)
        features = self.getFeaturesOffense(gameState, action)

        reward = self.getReward(gameState, action)
        maxQ = self.getMaxQ(nextState)
        q = self.evl2(gameState, Directions.STOP)

        for f in features:
            print "feature and weight:", f, features[f]

            self.weights[f] += alpha * (reward + discount * maxQ - q) * features[f]
            print f, self.weights[f]


class DefensiveReflexAgent(ReflexCaptureAgent):

    #################
    # help function #
    #################

    def atCenter(self, myPos):
        if myPos in self.boundary:
            return True
        else:
            return False

    # return a list of (x,y), shows the enemies positions
    def getEnemy(self, gameState):
        enemyPos = []
        enemies = self.getOpponents(gameState)
        for enemy in enemies:
            pos = gameState.getAgentPosition(enemy)
            if pos != None:
                enemyPos.append((enemy, pos))

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
        self.currFoodNum = len(currentFood.asList())

        previousFood = None
        if previousState != None:
            previousFood = self.getFoodYouAreDefending(previousState)
            self.preFoodNum = len(previousFood.asList())
            if not currentFood == previousFood and self.currFoodNum < self.preFoodNum:
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

        for i in range(len(list_pre)):
            if not list_pre[i] == list_curr[i]:
                for j in range(len(list_curr[i])):
                    if not list_curr[i][j] == list_pre[i][j]:
                        foodPos.append((i, j))

        return foodPos

    def evaluate(self, gameState, action, agentType):

        """
        Computes a linear combination of features and feature weights
        """
        return {
            'inital': self.getFeatureInital(gameState, action) * self.getWeightsInital(gameState, action),
            'defence': self.getFeaturesDefence(gameState, action) * self.getWeightsDefence(gameState, action),
            'offence': self.getFeaturesOffense(gameState, action) * self.getWeightOffence(gameState, action),
        }.get(agentType)



    def getWeightOffence(self, gameState, action):

        return {'score': 1.78261354182, 'DisToNearestFood': -4.91094492098, 'disToGhost': 8.17572535548,
                'disToCapsule': -1.36111562824, 'dots': -0.877933155097,
                'disToBoundary': -2.94156916302, 'deadends': -10}

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

        return {'disToBoundary': 100}

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
        if (enemyDistance <= 5):
            features['danger'] = 1
            if (enemyDistance <= 1 and self.ScaredTimer(successor) > 0):
                features['danger'] = -1
        else:
            features['danger'] = 0

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        # get the distance where the food lost
        if not self.foodLostPosition() == None:
            food = self.foodLostPosition()[0]
            # food = food[0]
            minDistance = self.getMazeDistance(myPos, food)

            self.foodLost = []
            self.foodLost.append(minDistance)
            if features['invaderDistance'] > 0:
                features['lostFoodDistance'] = 0
            else:
                features['lostFoodDistance'] = self.foodLost[0]

        # disToBoundary = 99999
        # for a in range(len(self.boundary)):
        #     disToBoundary = min(disToBoundary, self.getMazeDistance(myPos, self.boundary[a]))
        #     if myState.isPacman:
        #         features['distToBoundary'] = disToBoundary
        #     else:
        #         features['distToBoundary'] = 0

        return features

    def getWeightsDefence(self, gameState, action):
        return {'numInvaders': -1000, 'onDefense': 1, 'invaderDistance': -10000, 'stop': -1, 'reverse': -1, 'danger': 1
            , 'lostFoodDistance': -100,}

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        # get all legal actions
        actions = gameState.getLegalActions(self.index)
        # sorry here bro
        actions.remove(Directions.STOP)
        myPos = gameState.getAgentPosition(self.index)
        opponents = self.getOpponents(gameState)

        """
        switch agent type here
        """
        # there is no enemy, try to catch some food
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

    def __str__(self):
        print "Depth:", self.depth, "\nSubtrees:", len(self.subtrees), \
            "\nVisited:", str(self.visited), "\nScore:", str(self.score)
        return ""

    def expand(self):
        # print "Expanding..."
        if len(self.subtrees) == 0:
            for action in self.gameState.getLegalActions(self.agentindex):
                self.subtrees[action] = MCTree(self.gameState.generateSuccessor(self.agentindex, action), self,
                                               self.agent)
            print "expanded tree:", self
            # self.subtreesCounter[action] = 0
            # print "Expanded"
        # else:
        # print "No expand"
        # else:

#     def tree_policy(self):
#         actions = self.gameState.getLegalActions(self.agentindex)
#         if len(actions) == 1:
#             return actions[0]
#
#         maxQ = -10000
#         maxQaction = Directions.STOP
#         CP = 1 / 2
#         for action in actions:
#             # if self.subtreesCounter[action] == 0:
#             #     self.subtreesCounter[action] += 1
#             if self.subtrees[action].visited == 0:
#                 print "Tree policy returning new:", action
#                 return action
#             uct = self.agent.evl(self.gameState.generateSuccessor(self.agentindex, action)) + 2 * CP * sqrt(
#                 2 * log(self.visited) / self.subtrees[action].visited)
#             if uct >= maxQ:
#                 maxQ = uct
#                 maxQaction = action
#         # self.subtreesCounter[action] += 1
#         print "Tree policy returning: ", maxQaction
#         return maxQaction
#
#     def backprop(self, simulate_score):
#         self.visited += 1
#         self.score += simulate_score
#         if self.ancestor is not None:
#             print "from depth", self.depth
#             print "backprop to ancestor: ", simulate_score
#             self.ancestor.backprop(simulate_score)
#         else:
#             print "Backproped to root"
#             print "the depth is:", self.depth
#             print "root socre:", self.score
#
#
# def random_simulation(gameState, agent, depth):
#     print "agent in simulation", agent
#     agentindex = agent.index
#     if depth > 0:
#         actions = gameState.getLegalActions(agentindex)
#         reverse_direction = Directions.REVERSE[gameState.getAgentState(agentindex).getDirection()]
#         if len(actions) > 1:
#             # print "Actions before remove: ", actions
#             # for action in actions:
#             #     print type(action)
#             # print "Reverse direction: ", reverse_direction, type(reverse_direction)
#             actions.remove(reverse_direction)
#             # print "Actions: ", actions
#             action = random.choice(actions)
#             # print "Action taken:", action
#         else:
#             action = actions[0]
#         random_simulation(gameState.generateSuccessor(agentindex, action), agent, depth - 1)
#     return agent.evl(gameState)
#
#
# def MCTsearch(gameState, agent, depth):
#     start_time = time.time()
#     print "Starting Search time: ", start_time
#     root = MCTree(gameState, None, agent)
#     root.expand()
#
#     while time.time() - start_time < 0.08:  # budget time
#         print "New simulation! Time Elapsed: ", time.time() - start_time
#         action = root.tree_policy()
#         tree = root.subtrees[action]
#         print "number of taking the action on root:", tree.visited
#         while not tree.visited == 0:
#             print "subtree depth:", tree.depth
#             # print tree
#             # print "Calling for expansion"
#             tree.expand()  # effective if len(subtrees) == 0
#             # print "expanded tree: ", tree
#             action = tree.tree_policy()
#             tree = tree.subtrees[action]
#         simulated_score = random_simulation(tree.gameState, agent, depth)
#         print "simulated score:", simulated_score
#         tree.backprop(simulated_score)
#         print "After backprop, root:", root
#
#     maxQ = -10000
#     maxQaction = None
#     for action in root.gameState.getLegalActions(root.agentindex):
#         subtree = root.subtrees[action]
#         if subtree.visited == 0:
#             print "Error, an action is never simulated:", action
#             continue
#         qvalue = subtree.score / subtree.visited
#         if qvalue >= maxQ:
#             maxQ = qvalue
#             maxQaction = action
#     print "********RETURN ACTION*********", maxQaction
#     print root
#     # exit(100)
#     return maxQaction

#
# def evl(gameState):
#     value = random.uniform(0,1)
#     # print "random value:", value
#     return value


#
# class myoffensiveagent(baselineTeam.ReflexCaptureAgent):
#     def chooseAction(self, gameState):
#         return MCTsearch(gameState, self.index, 5)
