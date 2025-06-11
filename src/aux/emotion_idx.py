emotion_to_index_mma = {
            "angry": 0,
            "disgust": 1,
            "fear": 2,
            "happiness": 3,
            "sadness": 5,
            "surprise": 6,
            "neutral": 4
        }

index_to_emotion_mma = {v: k for k, v in emotion_to_index_mma.items()}

emotion_to_index_sfew = {
            "angry": 0,
            "disgust": 1,
            "fear": 2,
            "happiness": 3,
            "sadness": 5,
            "surprise": 6,
            "neutral": 4
        }

index_to_emotion_sfew = {v: k for k, v in emotion_to_index_sfew.items()}

emotion_to_index_aff = {
            "surprise": 3,
            "fear": 4,
            "disgust": 5,
            "happiness": 1,
            "sadness": 2,
            "angry": 6,
            "neutral": 0
        }

index_to_emotion_aff = {v: k for k, v in emotion_to_index_aff.items()}

emotion_to_index_raf = {
            "surprise": 0,
            "fear": 1,
            "disgust": 2,
            "happiness": 3,
            "sadness": 4,
            "angry": 5,
            "neutral": 6
        }

index_to_emotion_raf = {v: k for k, v in emotion_to_index_raf.items()}

emotion_to_index_fer = {
            "neutral": 0,
            "happiness": 1,
            "surprise": 2,
            "sadness": 3,
            "angry": 4,
            "disgust": 5,
            "fear": 6
        }

index_to_emotion_fer = {v: k for k, v in emotion_to_index_fer.items()}