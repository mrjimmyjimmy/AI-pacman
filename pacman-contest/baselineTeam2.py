

from captureAgents import CaptureAgent
import random, time, util
from game import Directions
from util import nearestPoint



from math import sqrt, log
import random, time

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent'):
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """
    def __init__(self, gameState):
        CaptureAgent.__init__(self, gameState)

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


    def foodLostPosition(self):
        currentState = self.getCurrentObservation()
        previousState = self.getPreviousObservation()
        currentFood = self.getFoodYouAreDefending(currentState)
        previousFood = None
        if previousState != None:
            previousFood = self.getFoodYouAreDefending(previousState)
            if not currentFood == previousFood:
                food = self.findLostFood(currentFood, previousFood)
                return food


    def findLostFood(self, current, previous):

        list_curr = []
        list_pre = []
        foodPos = []
        for i in current:
            list_curr.append(i)
            # list_curr.reverse()
        for i in previous:
            list_pre.append(i)
            # list_pre.reverse()

        for i in range (len(list_pre)):
            if not list_pre[i] == list_curr[i]:
                for j in range(len(list_curr[i])):
                    if not list_curr[i][j] == list_pre[i][j]:
                        foodPos.append((i,j))

        return foodPos

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

    def getFeatureInital(self, gameState, action):

        # inital features
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        # get the position
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        disToBoundary = min(self.getMazeDistance(myPos, self.boundary))
        features['distToBoundary'] = disToBoundary

        return features

    def getWeightInital(self, gameState, action):

        return {'disToBoundary' : 1}




class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """
    foodLost = float
    def getFeatures(self, gameState, action):

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

        noisyD = gameState.getAgentDistances()
        print noisyD


        return features

    def getWeights(self, gameState, action):
        return {'numInvaders': -100, 'onDefense': 1, 'invaderDistance': -100, 'stop': -1, 'reverse': -1, 'danger': 1
            , 'lostFoodDistance': -10}



    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = gameState.getLegalActions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(gameState, a) for a in actions]

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        foodLeft = len(self.getFood(gameState).asList())

        if foodLeft <= 0:
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction

        return random.choice(bestActions)


class OffensiveReflexAgent(ReflexCaptureAgent):

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        return MCTsearch(gameState, self, depth=5)

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
        features['dots'] = gameState.getAgentState(self.index).numCarrying
        # ---------------------feature 6: dis to boundary----------------
        disToBoundary = 99999
        for a in range(len(self.boundary)):
            disToBoundary = min(disToBoundary, self.getMazeDistance(agentPosition, self.boundary[a]))
        features['disToBoundary'] = disToBoundary
        # ---------------------feature 7: dis to opponent's attackers----------------
        #  need more work on this feature

        return features

    def getWeights(self, gameState):
        weights = {'score': 30, 'DisToNearestFood': -5, 'disToGhost': 50, 'disToCapsule': -55, 'dots': 50,
                   'disToBoundary': -50}
        #
        # enemies = []
        # for e in self.getOpponents(gameState):
        #     enemyState = gameState.getAgentState(e)
        #     if not enemyState.isPacman and not enemyState.getPosition() is None:
        #         enemies.append(enemyState)
        return weights



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
        return ""


    def expand(self):
        if len(self.subtrees) == 0:
            for action in self.gameState.getLegalActions(self.agentindex):
                self.subtrees[action] = MCTree(self.gameState.generateSuccessor(self.agentindex, action), self, self.agent)


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
        return maxQaction


    def backprop(self, simulate_score):
        self.visited += 1
        self.score += simulate_score
        if self.ancestor is not None:
            self.ancestor.backprop(simulate_score)


def random_simulation(gameState, agent, depth):
    agentindex = agent.index
    if depth > 0:
        actions = gameState.getLegalActions(agentindex)
        reverse_direction = Directions.REVERSE[gameState.getAgentState(agentindex).getDirection()]
        if len(actions) > 1:
            actions.remove(reverse_direction)
            action = random.choice(actions)
        else:
            action = actions[0]
        random_simulation(gameState.generateSuccessor(agentindex, action), agent, depth - 1)
    return agent.evl(gameState)


def MCTsearch(gameState, agent, depth):
    start_time = time.time()
    root = MCTree(gameState, None, agent)
    root.expand()

    while time.time() - start_time < 0.08:  # budget time
        action = root.tree_policy()
        tree = root.subtrees[action]
        while not tree.visited == 0:
            tree.expand()   # effective if len(subtrees) == 0
            action = tree.tree_policy()
            tree = tree.subtrees[action]
        simulated_score = random_simulation(tree.gameState, agent, depth)
        tree.backprop(simulated_score)

    maxQ = -10000
    maxQaction = None
    for action in root.gameState.getLegalActions(root.agentindex):
        subtree = root.subtrees[action]
        if subtree.visited == 0:
            continue
        qvalue = subtree.score / subtree.visited
        if qvalue >= maxQ:
            maxQ = qvalue
            maxQaction = action
    # exit(100)
    return maxQaction