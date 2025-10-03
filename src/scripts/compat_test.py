from manim import Scene, Line, Dot, always_redraw, DEGREES, YELLOW, WHITE, UP, ValueTracker, linear
import numpy as np

class CompatibilityTest(Scene):
    def construct(self):
        # Extend wait time to ensure duration shows >0s in common players (e.g., Windows Explorer)
        self.wait(2.0)


class PendulumTest(Scene):
    def construct(self):
        pivot = UP * 2.5
        L = 2.5
        amplitude = 30 * DEGREES
        period = 2.5
        omega = 2 * np.pi / period

        t = ValueTracker(0.0)

        def bob_point():
            theta = amplitude * np.cos(omega * t.get_value())
            return pivot + L * np.array([np.sin(theta), -np.cos(theta), 0])

        rod = always_redraw(lambda: Line(pivot, bob_point()))
        bob = always_redraw(lambda: Dot(bob_point(), radius=0.06, color=YELLOW))
        pivot_dot = Dot(pivot, radius=0.05, color=WHITE)

        self.add(rod, bob, pivot_dot)
        self.play(t.animate.set_value(20), run_time=20, rate_func=linear)
