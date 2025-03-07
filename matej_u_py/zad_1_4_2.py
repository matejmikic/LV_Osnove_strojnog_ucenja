
while 1:
    try:

        grade = input('Select a number from 0 to 1.0: ')

        grade = float(grade)

        if grade < 0.0 or grade > 1.0:
            raise ValueError('Selected number is not in the range')

        if not isinstance(grade,float):
            raise TypeError ("You didn't give me a number")


        if grade >= 0.9:
            print('Grade A')
        elif grade >= 0.8:
            print('Grade B')
        elif grade >= 0.7:
            print('Grade C')
        elif grade >= 0.6:
            print('Grade D')
        elif grade < 0.6:
            print('Grade F')
    except TypeError:
        print('NOOOOOO')
    except ValueError:
        print('NOOOOOO')
    else: 
        break