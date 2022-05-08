import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")


def has_go_token(doc):
    for t in doc:
        if t.lower_ in ['go', 'golang', 'python', 'ruby', 'objective-c']:
            if t.pos_ != 'VERB':
                return True
    return False


doc = nlp("i am an iOS dev and I like to code in objective-c")
# lower, is_punct, op, pos{in}
obj_c_pattern1 = [{'LOWER': 'objective'},
                  {'IS_PUNCT': True, 'OP': '?'},
                  {'LOWER': 'c'}]

obj_c_pattern2 = [{'LOWER': 'objectivec'}]

golang_pattern1 = [{'LOWER': 'golang'}]
golang_pattern2 = [{'LOWER': 'go',
                    'POS': {'NOT_IN': ['VERB']}}]

python_pattern = [{'LOWER': 'python'}]
ruby_pattern = [{'LOWER': 'ruby'}]
js_pattern = [{'LOWER': {'IN': ['js', 'javascript']}}]

matcher = Matcher(nlp.vocab, validate=True)
matcher.add("OBJ_C_LANG", None, obj_c_pattern1, obj_c_pattern2)
matcher.add("PYTHON_LANG", None, python_pattern)
matcher.add("GO_LANG", None, golang_pattern1, golang_pattern2)
matcher.add("JS_LANG", None, js_pattern)
matcher.add("RUBY_LANG", None, ruby_pattern)

doc = nlp("I am an iOS dev who codes in both python, go/golang as well as objective-c")
for match_id, start, end in matcher(doc):
    print(doc[start: end])
