import cv2 

def main():
    #Carrega arquivos necessarios
    XML_PATH = 'cascades/haarcascade_fullbody.xml'
    personCascade = cv2.CascadeClassifier(XML_PATH)
    video = cv2.VideoCapture('vtest.avi')

    # Verifica se o vídeo foi aberto corretamente
    if not video.isOpened():
        print("Erro ao abrir o vídeo.")
        exit()

    ret, frame = video.read()
    while ret:
        frame_show = frame
        
        #Ignora a informação de cores e detecta as pessoas
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = personCascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        #Desenha um retangulo verde envolta da pessoa detectada
        for (x, y, w, h) in detections:
            cv2.rectangle(frame_show, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
        #mostra a imagem e coleta o próximo frame
        cv2.imshow('Teste', frame_show)   
        ret, frame = video.read()

        #Interrompe o programa com a tecla ESC
        if cv2.waitKey(1) == 27:
            break

if __name__=="__main__":
    main()