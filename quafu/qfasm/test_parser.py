# Yacc example
 
import ply.yacc as yacc
 
# Get the token map from the lexer.  This is required.
from test_lexer import tokens
 
def p_expression_plus(p):
    """
    expression : expression PLUS term
                | expression PLUS error
    """
    if isinstance(p[3],int):
        p[0] = p[1] + p[3]
    else:
        print("type", type(p))
        print("value:",p)
        print("type", type(p[1]))
        print("value:",p[1])
        print("type", type(p[3]))
        print("value:",p[3].value)

def p_expression_minus(p):
    """
    expression : expression MINUS term
                | expression MINUS error 
    """
    print(type(p[3]))
    print(p[3])
    p[0] = p[1] - p[3]
 
def p_expression_term(p):
    'expression : term'
    p[0] = p[1]
 
def p_term_times(p):
    'term : term TIMES factor'
    p[0] = p[1] * p[3]
 
def p_term_div(p):
    'term : term DIVIDE factor'
    p[0] = p[1] / p[3]
 
def p_term_factor(p):
    'term : factor'
    p[0] = p[1]
 
def p_factor_num(p):
    'factor : NUMBER'
    # print(p[1])
    p[0] = p[1]
 
def p_factor_othe(p):
    'factor : OTHE'
    print(f"Type OTHE, VALUE {p[1]}")
    p[0] = p[1]

def p_factor_expr(p):
    """
    factor : LPAREN expression RPAREN
    """
    p[0] = p[2]
 
# Error rule for syntax errors
def p_error(p):
    print("Syntax error in input!")
 
# Build the parser
parser = yacc.yacc()

while True:
   try:
       s = input('calc > ')
   except EOFError:
       break
   if not s: continue
   result = parser.parse(s)
   print(result)
