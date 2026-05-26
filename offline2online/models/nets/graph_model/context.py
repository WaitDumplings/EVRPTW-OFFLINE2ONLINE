"""
Problem specific global embedding for global context.
"""

import torch
from torch import nn


def AutoContext(problem_name, config):
    """
    Automatically select the corresponding module according to ``problem_name``
    """
    mapping = {
        "evrptw": VRPContext,
    }
    embeddingClass = mapping[problem_name]
    embedding = embeddingClass(**config)
    return embedding


def _gather_by_index(source, index):
    """
    target[i,1,:] = source[i,index[i],:]
    Inputs:
        source: [B x H x D]
        index: [B x 1] or [B]
    Outpus:
        target: [B x 1 x D]
    """
    target = torch.gather(source, 1, index.unsqueeze(-1).expand(-1, -1, source.size(-1)))
    return target


class PrevNodeContext(nn.Module):
    """
    Abstract class for Context.
    Any subclass, by default, will return a concatenation of

    +---------------------+-----------------+
    | prev_node_embedding | state_embedding |
    +---------------------+-----------------+

    The ``prev_node_embedding`` is the node embedding of the last visited node.
    It is obtained by ``_prev_node_embedding`` method.
    It requires ``state.get_current_node()`` to provide the index of the last visited node.

    The ``state_embedding`` is the global context we want to include, such as the remaining capacity in VRP.
    It is obtained by ``_state_embedding`` method.
    It is not implemented. The subclass of this abstract class needs to implement this method.

    Args:
        problem: an object defining the settings of the environment
        context_dim: the dimension of the output
    Inputs: embeddings, state
        * **embeddings** : [batch x graph size x embed dim]
        * **state**: An object providing observations in the environment. \
                    Needs to supply ``state.get_current_node()``
    Outputs: context_embedding
        * **context_embedding**: [batch x 1 x context_dim]

    """

    def __init__(self, context_dim):
        super(PrevNodeContext, self).__init__()
        self.context_dim = context_dim

    def _prev_node_embedding(self, embeddings, state):
        current_node = state.get_current_node()
        prev_node_embedding = _gather_by_index(embeddings, current_node)
        return prev_node_embedding

    def _state_embedding(self, embeddings, state):
        raise NotImplementedError("Please implement the embedding for your own problem.")

    def forward(self, embeddings, state):
        prev_node_embedding = self._prev_node_embedding(embeddings, state)
        # remaining loading & remaining battery
        state_embedding = self._state_embedding(embeddings, state)
        # Embedding of previous node + remaining capacity
        context_embedding = torch.cat((prev_node_embedding, state_embedding), -1)
        return context_embedding

class VRPContext(PrevNodeContext):
    """
    Context node embedding for capacitated vehicle routing problem.
    Return a concatenation of

    +---------------------------+--------------------+
    | previous node's embedding | remaining capacity |
    +---------------------------+--------------------+

    .. note::
        Subclass of :class:`.PrevNodeContext`. The argument, inputs, outputs follow the same specification.

        In addition to supplying  ``state.get_current_node()`` for the index of the previous visited node.
        The input ``state`` needs to supply ``state.VEHICLE_CAPACITY`` and ``state.used_capacity``
        for calculating the remaining capcacity.
    """

    def __init__(self, context_dim):
        super(VRPContext, self).__init__(context_dim)

    def _state_embedding(self, embeddings, state):
        state_embedding_capacity = (state.used_capacity).unsqueeze(-1)
        state_embedding_battery  = (state.used_battery).unsqueeze(-1)
        state_embedding_time = state.current_time.unsqueeze(-1)
        state_visited_customers_ratio = state.visited_customers_ratio
        state_remain_feasible_customers_ratio = state.remain_feasible_customers_ratio
        return torch.cat((state_embedding_capacity, state_embedding_battery, state_embedding_time, state_visited_customers_ratio, state_remain_feasible_customers_ratio), dim=-1)


