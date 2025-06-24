from facenet_pytorch import InceptionResnetV1

def load_model():
    model = InceptionResnetV1(pretrained='vggface2').eval()
    return model