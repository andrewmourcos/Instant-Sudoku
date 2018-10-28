# 48->57
import cv2
import os

for i in range(1,3080):
	try:
		i = str(i)
		path1 = 'Acquisition/DigitExample/'+i+'.jpg'
		img = cv2.imread(path1)
		cv2.imshow('img',img)

		k = cv2.waitKey(0)
		if k==27:
			print(i) # print last image number
			cv2.destroyAllWindows()

		elif k==-1:  # normally -1 returned,so don't print it
			pass
		elif k==48:
			os.rename(path1, "Train/0/"+i+".jpg")
			cv2.destroyAllWindows()

		elif k==49:
			os.rename(path1, "Train/1/"+i+".jpg")
			cv2.destroyAllWindows()

		elif k==50:
			os.rename(path1, "Train/2/"+i+".jpg")
			cv2.destroyAllWindows()

		elif k==51:
			os.rename(path1, "Train/3/"+i+".jpg")
			cv2.destroyAllWindows()

		elif k==52:
			os.rename(path1, "Train/4/"+i+".jpg")
			cv2.destroyAllWindows()

		elif k==53:
			os.rename(path1, "Train/5/"+i+".jpg")
			cv2.destroyAllWindows()

		elif k==54:
			os.rename(path1, "Train/6/"+i+".jpg")
			cv2.destroyAllWindows()

		elif k==55:
			os.rename(path1, "Train/7/"+i+".jpg")
			cv2.destroyAllWindows()

		elif k==56:
			os.rename(path1, "Train/8/"+i+".jpg")
			cv2.destroyAllWindows()

		elif k==57:
			os.rename(path1, "Train/9/"+i+".jpg")
			cv2.destroyAllWindows()
	except:
		pass


