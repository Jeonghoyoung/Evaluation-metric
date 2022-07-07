from konlpy.tag import Okt

okt = Okt()
t = '안녕하세요 여기는 어디인가요?'
print(okt.morphs(t))