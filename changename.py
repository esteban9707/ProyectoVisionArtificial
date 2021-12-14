import os
for dir in [1,2]:
    path = "C:\\Users\\HP\\Desktop\\Universidad\\9 semestre\\Inteligentes\\Dataset billetes"
    path = os.path.join(path,str(dir))
    i=1
    for filename in os.listdir(path):
        new_name = os.path.join(path,str(dir)+'_'+str(i)+'.jpg')
        os.rename(os.path.join(path,filename), new_name)
        print(new_name)
        i=i+1