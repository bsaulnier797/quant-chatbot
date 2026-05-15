with open("app.py", "r") as f:
    content = f.read()
    start = content.find("def try_render_chart")
    print(content[start:start+1500])
