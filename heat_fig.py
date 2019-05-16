# -*- coding: UTF-8 -*-

#
# print('\n'.join([''.join([('GoodLuck 2019'[(x-y)%13]
# if((x*0.05)**2+(y*0.1)**2-1)**3-(x*0.05)**2*(y*0.1)**3 <= 0
#     else' ')
# for x in range(-30, 30)])
# for y in range(15, -15, -1)]))

with open('/home/xh/test.txt', 'w') as f:
    f.write('\n'.join([''.join([('U'[(x-y)%1]
if((x*0.05)**2+(y*0.1)**2-1)**3-(x*0.05)**2*(y*0.1)**3 <= 0
    else' ')
for x in range(-30, 30)])
for y in range(15, -15, -1)]))