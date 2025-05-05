import os
print("Hello World!")
if not os.path.exists("prova"):
        print("MKDIR")
        os.mkdir("prova")