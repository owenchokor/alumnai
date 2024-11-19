def vps_idenfier(string):
    stack = []
    for parentheses in string:
        if parentheses =="(":
            stack.append(parentheses)
        else:
            if len(stack):
                stack.pop()
            else:
                return "NO"
    if len(stack):
        return "NO"
    else:
        return "YES"

lines = int(input())

for i in range(lines):
    line = input()
    print(vps_idenfier(line))