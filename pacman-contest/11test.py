# ###########
# # min-max #
# ###########
#
# def evaluate_minimax(self, gameState, agentType):
#     actions = gameState.getLegalActions(self.index)
#     values = [self.evaluate(gameState, a, agentType) for a in actions]
#     maxValue = max(values)
#     return maxValue
#
# def minMaxValue(self, gameState, agentIndex, alpha, beta, depth, agentType):
#     if depth == 0 or gameState.isOver():
#         return self.evaluate_minimax(gameState, agentType)
#
#     if agentIndex == self.index:
#         return self.maxValue(gameState, agentIndex, alpha, beta, depth, agentType)
#
#     elif gameState.isOnRedTeam(agentIndex) != gameState.isOnRedTeam(self.index) \
#             and gameState.getAgentPosition(agentIndex) != None \
#             and not gameState.getAgentState(agentIndex).isPacman:
#         return self.minValue(gameState, agentIndex, alpha, beta, depth, agentType)
#     else:
#         return self.minMaxValue(gameState, (agentIndex + 1) % 4, alpha, beta, depth, agentType)
#
# def maxValue(self, gameState, agentIndex, alpha, beta, depth, agentType):
#     v = -float('inf')
#     for a in gameState.getLegalActions(agentIndex):
#         s = gameState.generateSuccessor(agentIndex, a)
#         v = max(v, self.minMaxValue(s, ((agentIndex + 1) % 4), alpha, beta, depth - 1, agentType))
#         if v >= beta:
#             return v
#         alpha = max(alpha, v)
#     return v
#
# def minValue(self, gameState, agentIndex, alpha, beta, depth, agentType):
#     v = float('inf')
#     past_dist = self.getMazeDistance(gameState.getAgentPosition(self.index), gameState.getAgentPosition(agentIndex))
#     for a in gameState.getLegalActions(agentIndex):
#         s = gameState.generateSuccessor(agentIndex, a)
#         if self.getMazeDistance(s.getAgentPosition(self.index), gameState.getAgentPosition(self.index)) >= 2 and s.getAgentPosition(self.index) == s.getInitialAgentPosition(self.index):
#             return -float('inf')
#         if self.getMazeDistance(s.getAgentPosition(self.index),
#                                 s.getAgentPosition(agentIndex)) < past_dist or self.getMazeDistance(s.getAgentPosition(self.index),
#                                                                                                     s.getAgentPosition(agentIndex)) <= 2:
#             v = min(v, self.minMaxValue(s, ((agentIndex + 1) % 4), alpha, beta, depth - 1, agentType))
#             if v <= alpha:
#                 return v
#             beta = min(beta, v)
#     return v
