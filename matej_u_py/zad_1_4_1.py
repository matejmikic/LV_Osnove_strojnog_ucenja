radni_sati = input('Radni sati: ')
satnica = input('eura/h: ')

radni_sati = float(radni_sati)
satnica = float(satnica)


plaća = radni_sati * satnica

print('Zarađeno je ' ,plaća , 'eura')


def total_euro(radni_sati,satnica):
    plaća = radni_sati * satnica
    print('Zarađeno je ' , plaća , 'eura')