import tkinter as tk
import random
import fractions

TOTAL_QUESTIONS = 80
TIME_LIMIT = 8 * 60  # seconds

def random_mixed_fraction(val):
    """Convert float to a string like '2 1/2'."""
    frac = fractions.Fraction(val).limit_denominator(4)
    if frac.denominator == 1:
        return str(frac.numerator)
    whole = frac.numerator // frac.denominator
    remainder = frac.numerator % frac.denominator
    if whole == 0:
        return f"{remainder}/{frac.denominator}"
    return f"{whole} {remainder}/{frac.denominator}"

def format_number(n):
    """Format float as string or mixed fraction."""
    if isinstance(n, int) or n == int(n):
        return str(int(n))
    elif random.random() < 0.5:
        return f"{n:.2f}"
    else:
        return random_mixed_fraction(n)

def parse_input(text):
    """Convert mixed fractions or decimals to float."""
    try:
        if " " in text:
            whole, frac = text.split()
            num, denom = frac.split('/')
            return int(whole) + int(num) / int(denom)
        elif "/" in text:
            num, denom = text.split('/')
            return int(num) / int(denom)
        else:
            return float(text)
    except:
        return None

def generate_question():
    op = random.choice(["+", "-", "*", "/"])
    reverse = random.choice([True, False])
    rhs_first = random.choice([True, False])
    is_fractional = False  # Flag to allow fractional answers

    if op == "+":
        a, b = random.randint(1, 99), random.randint(1, 99)
        correct = a + b
        if reverse:
            question = f"? + {b} = {correct}"
            ans = a
        elif rhs_first:
            question = f"? = {a} + {b}"
            ans = correct
        else:
            question = f"{a} + {b} = ?"
            ans = correct

    elif op == "-":
        a, b = random.randint(20, 99), random.randint(1, 20)
        correct = a - b
        if reverse:
            question = f"? - {b} = {correct}"
            ans = a
        elif rhs_first:
            question = f"? = {a} - {b}"
            ans = correct
        else:
            question = f"{a} - {b} = ?"
            ans = correct

    elif op == "*":
        a, b = random.randint(2, 20), random.randint(2, 12)
        correct = a * b
        if reverse:
            question = f"{a} × ? = {correct}"
            ans = b
        elif rhs_first:
            question = f"? = {a} × {b}"
            ans = correct
        else:
            question = f"{a} × {b} = ?"
            ans = correct

    elif op == "/":
        b = random.randint(2, 12)
        ans = round(random.uniform(1, 12), 2)
        a = round(b * ans, 2)
        correct = round(a / b, 2)
        is_fractional = True
        if reverse:
            question = f"{a} ÷ ? = {ans}"
            ans = b
        elif rhs_first:
            question = f"? = {a} ÷ {b}"
            ans = correct
        else:
            question = f"{a} ÷ {b} = ?"
            ans = correct

    ans = round(float(ans), 2)
    is_fractional = is_fractional or not ans.is_integer()

    # Safely generate 3 wrong answers + 1 correct
    choices = set()
    choices.add(ans)
    while len(choices) < 4:
        delta = random.uniform(1.0, 5.0) if is_fractional else random.randint(1, 10)
        wrong = round(ans + random.choice([-delta, delta]), 2)
        if abs(wrong - ans) > 0.05:
            choices.add(wrong)

    choices = list(choices)
    random.shuffle(choices)

    # Format for display
    if is_fractional:
        formatted_choices = [format_number(c) for c in choices]
    else:
        formatted_choices = [str(int(round(c))) for c in choices]

    return question, ans, formatted_choices

class OptiverApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Optiver 80-in-8 MCQ Test")
        self.score = 0
        self.current_q = 0
        self.correct_ans = None
        self.time_left = TIME_LIMIT

        self.question_label = tk.Label(root, text="Press Start", font=("Helvetica", 22), wraplength=450)
        self.question_label.pack(pady=20)

        self.buttons = []
        for i in range(4):
            btn = tk.Button(root, text="", font=("Helvetica", 18), width=20, command=lambda i=i: self.check_answer(i))
            btn.pack(pady=5)
            self.buttons.append(btn)

        self.feedback_label = tk.Label(root, text="", font=("Helvetica", 16))
        self.feedback_label.pack(pady=5)

        self.score_label = tk.Label(root, text="Score: 0", font=("Helvetica", 14))
        self.score_label.pack()

        self.timer_label = tk.Label(root, text="Time Left: 08:00", font=("Helvetica", 14))
        self.timer_label.pack()

        self.start_button = tk.Button(root, text="Start", font=("Helvetica", 16), command=self.start)
        self.start_button.pack(pady=10)

    def start(self):
        self.start_button.config(state=tk.DISABLED)
        self.root.after(1000, self.update_timer)
        self.next_question()

    def update_timer(self):
        if self.time_left > 0:
            mins, secs = divmod(self.time_left, 60)
            self.timer_label.config(text=f"Time Left: {mins:02}:{secs:02}")
            self.time_left -= 1
            self.root.after(1000, self.update_timer)
        else:
            self.end_test()

    def next_question(self):
        if self.current_q >= TOTAL_QUESTIONS:
            self.end_test()
            return
        self.current_q += 1
        q, ans, choices = generate_question()
        self.correct_ans = round(ans, 2)
        self.question_label.config(text=f"Q{self.current_q}: {q}")
        for i, choice in enumerate(choices):
            self.buttons[i].config(text=str(choice), state=tk.NORMAL)
        self.feedback_label.config(text="")

    def check_answer(self, idx):
        selected_text = self.buttons[idx].cget("text")
        selected_value = parse_input(selected_text)
        for btn in self.buttons:
            btn.config(state=tk.DISABLED)
        if selected_value is not None and abs(selected_value - self.correct_ans) < 1e-2:
            self.score += 1
            self.feedback_label.config(text="✅ Correct!", fg="green")
        else:
            self.feedback_label.config(text=f"❌ Wrong! Ans: {self.correct_ans}", fg="red")
        self.score_label.config(text=f"Score: {self.score}")
        self.root.after(700, self.next_question)

    def end_test(self):
        self.question_label.config(text="Time's up or all questions done!")
        for btn in self.buttons:
            btn.config(state=tk.DISABLED)
        self.feedback_label.config(text=f"Final Score: {self.score}/{self.current_q}", fg="blue")

if __name__ == "__main__":
    root = tk.Tk()
    app = OptiverApp(root)
    root.geometry("500x600")
    root.mainloop()