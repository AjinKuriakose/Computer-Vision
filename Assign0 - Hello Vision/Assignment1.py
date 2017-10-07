import cv2

filepath = r"C:\Users\ajinkuriakose\Desktop\Computer Vision\Assignments\Assignment0\bean.jpg"
img = cv2.imread(filepath)
scalar = 2
print img

cv2.imshow('raw_image', img)
k = cv2.waitKey(0)
cv2.destroyWindow('raw_image')
b, g, r = cv2.split(img)

# # Add a scalar to the image and display the result
addb = cv2.add(b, scalar)
addg = cv2.add(g, scalar)
addr = cv2.add(r, scalar)
addimg = cv2.merge((addb, addg, addr))
cv2.imshow('add_image', addimg)
cv2.waitKey(0)
cv2.destroyWindow('add_image')

# subtract a scalar to the image and display the result
subb = cv2.subtract(b, scalar)
subg = cv2.subtract(g, scalar)
subr = cv2.subtract(r, scalar)
subimg = cv2.merge((subb, subg, subr))
cv2.imshow('sub_image', subimg)
cv2.waitKey(0)
cv2.destroyWindow('sub_image')

# multiply each pixel with a scalar and display
mulb = cv2.multiply(b, scalar)
mulg = cv2.multiply(g, scalar)
mulr = cv2.multiply(r, scalar)
mulimg = cv2.merge((mulb, mulg, mulr))
cv2.imshow('mul_image', mulimg)
cv2.waitKey(0)
cv2.destroyWindow('mul_image')

# #Divide each pixel with a scalar and display
divb = cv2.divide(b, scalar)
divg = cv2.divide(g, scalar)
divr = cv2.divide(r, scalar)
divimg = cv2.merge((divb, divg, divr))
cv2.imshow('div_image', divimg)
cv2.waitKey(0)
cv2.destroyWindow('div_image')

# resize the image by 0.5
resizedimg = cv2.resize(img, None, fx=0.5, fy=0.5)
cv2.imshow('Resized_image', resizedimg)
cv2.waitKey(0)
cv2.destroyWindow('Resized_image')
