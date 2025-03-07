nums = []
print('Provide me with numbers')



while 1:
    try:
        num = input()
        if(num == 'Done'):
            break
        nums.append(float(num))
    except:
        print('that is not a number')

nums.sort()
print(nums)



print('You have entered ', len(nums), 'numbers')

print('Minimal element is ', min(nums))
print('Maximal element is ', max(nums))

total_sum = 0.0

for val in nums:
    total_sum = total_sum + val

avg = total_sum/len(nums)

print('Numbers mean is ', avg)


        

        
    
