from dataclasses import dataclass, field
from typing import List, Optional
import spacy

@dataclass
class Tok:
    type: str   # "WORD", "NUMBER", "SYMBOL"
    value: str

class DFATokenizer:
    def __init__(self):
        self.state = "START"
        self.current = ""
        self.tokens: List[Tok] = []

    def reset(self):
        self.state = "START"
        self.current = ""
        self.tokens.clear()

    def emit(self, type_):
        if self.current:
            self.tokens.append(Tok(type_, self.current))
            self.current = ""

    def tokenize(self, text: str) -> List[Tok]:
        self.reset()
        for ch in text:
            if self.state == "START":
                if ch.isspace():
                    continue
                elif ch.isalpha():
                    self.state = "WORD"
                    self.current += ch
                elif ch.isdigit():
                    self.state = "NUMBER"
                    self.current += ch
                else:
                    self.tokens.append(Tok("SYMBOL", ch))

            elif self.state == "WORD":
                if ch.isalpha() or ch == "'":
                    self.current += ch
                else:
                    self.emit("WORD")
                    self.state = "START"
                    # reprocess ch
                    if ch.isspace():
                        continue
                    elif ch.isdigit():
                        self.state = "NUMBER"
                        self.current += ch
                    else:
                        self.tokens.append(Tok("SYMBOL", ch))

            elif self.state == "NUMBER":
                if ch.isdigit():
                    self.current += ch
                else:
                    self.emit("NUMBER")
                    self.state = "START"
                    # reprocess ch
                    if ch.isspace():
                        continue
                    elif ch.isalpha():
                        self.state = "WORD"
                        self.current += ch
                    else:
                        self.tokens.append(Tok("SYMBOL", ch))

        if self.state == "WORD":
            self.emit("WORD")
        elif self.state == "NUMBER":
            self.emit("NUMBER")

        # normalize words to lowercase for convenience
        norm = []
        for t in self.tokens:
            if t.type == "WORD":
                norm.append(Tok("WORD", t.value.lower()))
            else:
                norm.append(t)
        return norm

@dataclass
class DepNode:
    text: str
    lemma: str
    pos: str
    dep: str
    children: List["DepNode"] = field(default_factory=list)

    def pretty(self, level=0) -> str:
        indent = "  " * level
        out = f"{indent}{self.text} ({self.pos}, {self.dep})\n"
        for child in self.children:
            out += child.pretty(level + 1)
        return out


def build_dep_tree(doc) -> Optional[DepNode]:
    if len(doc) == 0:
        return None

    # Create nodes for each token
    nodes = [DepNode(t.text, t.lemma_, t.pos_, t.dep_) for t in doc]

    root = None
    for i, tok in enumerate(doc):
        if tok.head.i == tok.i:
            # This is the ROOT token
            root = nodes[i]
        else:
            head_node = nodes[tok.head.i]
            head_node.children.append(nodes[i])

    return root

def is_likely_natural_sentence(doc) -> bool:
    #exactly one ROOT, ROOT is a verb or auxiliar, at least one nominal subject
 #uusing spacy learned grammar as a reference, not defining our cfg 

    roots = [t for t in doc if t.dep_ == "ROOT"]
    if len(roots) != 1:
        return False

    root = roots[0]
    if root.pos_ not in ("VERB", "AUX"):
        return False

    has_subject = any(t.dep_ in ("nsubj", "nsubjpass", "csubj") for t in doc)
    if not has_subject:
        return False

    return True


def main():
    # load spacy model (predefined "grammar")
    # make sure you installed it first:
    #   python -m spacy download en_core_web_sm
    nlp = spacy.load("en_core_web_sm")

    sentences = [
        "The man saw a dog.",
        "cat table green quickly.",
        "I will build a parser using automata.",
        "x1 + x2 = 10"
    ]

    tokenizer = DFATokenizer()

    for s in sentences:
        print("=" * 60)
        print("Sentence:", s)

        #  low-level DFA tokenization
        my_tokens = tokenizer.tokenize(s)
        print("\nDFA tokens:")
        for t in my_tokens:
            print(" ", t)

        # spacy analysis
        doc = nlp(s)

        print("\nspaCy tokens / POS / dep (reference grammar):")
        for t in doc:
            print(f"  {t.i:2d}: {t.text:10s} POS={t.pos_:6s} DEP={t.dep_:10s} HEAD={t.head.i}")

        # Use spacy arcs to build our parse tree
        tree = build_dep_tree(doc)
        print("\nOur dependency tree (built manually):")
        if tree:
            print(tree.pretty().rstrip())
        else:
            print("  <empty>")

        print("\nLikely natural language sentence?",
              "YES" if is_likely_natural_sentence(doc) else "NO")


if __name__ == "__main__":
    main()

