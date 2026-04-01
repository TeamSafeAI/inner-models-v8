"""
engine/ — Modular brain simulation.

Usage:
    from engine.loader import load
    from engine.runner import Brain

    brain_data = load('brains/my_brain.db')
    brain = Brain(brain_data, learn=True)
    fired = brain.tick()                     # one tick
    recorder = brain.run(10000)              # or run N ticks
    recorder.report()                        # see what happened
"""
