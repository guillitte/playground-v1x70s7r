# Bonjour!

Ce programme python illustre l'algorithme de descente de gradient pour rechercher un minimum local d'une fonction f d'une variable réelle.  
Son principe est simple : pour trouver le minimum, il suffit d'effectuer des petits pas dans le sens de la descente, juqu'au momment où la pente s'annule.\ 
Nous rechercherons un minimum de la fonction f telle que f(x) = (x-1)²+(1/x-2)². 
Pour une fonction d'une variable, le gradient est simplement la dérivée : f'(x) = 2(x-1)+2(1/x-2)(-1/x²) 
Ce programme comporte deux versions de l'agorithme : 
* La première utilise la dérivée analytique de f, qui doit lui être fournie
* La seconde utilise une approximation numérique de cette dérivée  


```python runnable
def f(x):
    return (x-1)**2+(1/x-2)**2

def df(x):
    return 2*(x-1) + 2*(1/x-2)*(-1/x**2)


def descente1(f, df, a, alpha=1e-2, eps=1e-9, maxIter=1000):
    # Recherche le minimum d'une fonction f par descente de gradient
    # df doit être la dérivée de f
    # a est la valeur initiale
    # alpha est le taux d'apprentissage qui détermine la rapidité de la descente (par défaut 1/100)
    # eps est la précision souhaitée (par défaut 1/1000000000)
    # maxIter est le nombre maximum d'itération (par défaut 1000)
    
    grad = df(a)
    i=0
    while abs(grad)>eps: # tant que la pente n'est pas approximativement nulle
        grad = df(a) # on calcule la pente
        a = a-alpha*grad # on effectue un petit pas vers le bas
        i += 1
        #print(i, a, grad) # décommenter cette ligne pour imprimer les itérations
        if i > maxIter:
            return None # on abandonne si le nombre d'itérations est trop élevé
    return a

def descente2(f, a, lr=1e-2, eps=1e-9, maxIter=1000):
    # Recherche le minimum d'une fonction f par descente de gradient avec dérivée numérique
    # a est la valeur initiale
    # alpha est le taux d'apprentissage qui détermine la rapidité de la descente (par défaut 1/100)
    # eps est la précision souhaitée (par défaut 1/1000000000)
    # maxIter est le nombre maximum d'itération (par défaut 1000)    
    
    grad = 1
    i=0
    while abs(grad)>eps:
        grad = (f(a+eps)-f(a-eps))/(2*eps) #approximation numérique de la dérivée
        a = a-lr*grad
        i += 1
        #print(i, a, grad) # décommenter cette ligne pour imprimer les itérations
        if i > maxIter:
            return None
    return a



print(descente1(f, df, 1))
print(descente2(f, 1))


```

# Exercice

Comment transformer cette decente de gradient en montée, pour rechercher un maximum local ?
