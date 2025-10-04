from manim import Scene, Text, YELLOW, BLUE_E, TEAL_E, GREEN, ORANGE, WHITE


class FontShowcase(Scene):
    def construct(self):
        # 1) 通用字体轮播：每种字体展示 1 秒
        fonts = [
            "Ubuntu",
            "Inconsolata",
            "Source Code Pro",
            "DejaVu Sans Mono",
        ]
        colors = [WHITE, YELLOW, BLUE_E, TEAL_E, GREEN, ORANGE]

        for i, fam in enumerate(fonts):
            color = colors[i % len(colors)]
            t = Text(fam, font=fam, color=color).scale(1.3)
            self.add(t)
            self.wait(1.0)
            self.remove(t)

        # 2) 推荐组合（参考 3b1b 风格倾向：深底亮色、清晰的无衬线/等宽）
        combos = [
            ("Ubuntu", YELLOW),
            ("Source Code Pro", BLUE_E),
            ("Inconsolata", TEAL_E),
            ("DejaVu Sans Mono", GREEN),
        ]
        for fam, color in combos:
            t = Text(fam, font=fam, color=color).scale(1.3)
            self.add(t)
            self.wait(1.0)
            self.remove(t)
