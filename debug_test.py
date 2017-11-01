import os
import sys

parent = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(os.path.dirname(parent), "aimacode"))
import unittest
from aimacode.utils import expr
from aimacode.planning import Action
from example_have_cake import have_cake
from my_planning_graph import (
    PlanningGraph, PgNode_a, PgNode_s, mutexify
)

if __name__ == '__main__':
    p = have_cake()
    pg = PlanningGraph(p, p.initial)
    # some independent nodes for testing mutex
    na1 = PgNode_a(Action(expr('Go(here)'),
                                [[], []], [[expr('At(here)')], []]))
    na2 = PgNode_a(Action(expr('Go(there)'),
                                [[], []], [[expr('At(there)')], []]))
    na3 = PgNode_a(Action(expr('Noop(At(there))'),
                                [[expr('At(there)')], []], [[expr('At(there)')], []]))
    na4 = PgNode_a(Action(expr('Noop(At(here))'),
                                [[expr('At(here)')], []], [[expr('At(here)')], []]))
    na5 = PgNode_a(Action(expr('Reverse(At(here))'),
                                [[expr('At(here)')], []], [[], [expr('At(here)')]]))
    ns1 = PgNode_s(expr('At(here)'), True)
    ns2 = PgNode_s(expr('At(there)'), True)
    ns3 = PgNode_s(expr('At(here)'), False)
    ns4 = PgNode_s(expr('At(there)'), False)
    na1.children.add(ns1)
    ns1.parents.add(na1)
    na2.children.add(ns2)
    ns2.parents.add(na2)
    na1.parents.add(ns3)
    na2.parents.add(ns4)

    # Non-competing action nodes incorrectly marked as mutex
    PlanningGraph.competing_needs_mutex(pg, na1, na2)

    mutexify(ns3, ns4)

    # Opposite preconditions from two action nodes not marked as mutex
    PlanningGraph.competing_needs_mutex(pg, na1, na2)
    