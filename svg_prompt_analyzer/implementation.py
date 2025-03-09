from svg_prompt_analyzer import SVGPromptAnalyzerApp

app1 = SVGPromptAnalyzerApp("train.csv", "output/train")
results = app1.run()

app2 = SVGPromptAnalyzerApp("test.csv", "output/test")
results = app2.run()