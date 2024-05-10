# Import related library
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Function untuk prediksi data
def prediction(file, chosen_model, df_kalori):
    img = tf.keras.utils.load_img(file, target_size=(260, 260))
    x = tf.keras.utils.img_to_array(img)/255

    plt.imshow(img)

    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = chosen_model.predict(images, batch_size=10)
    idx = np.argmax(classes)
    clas = ['almonds','apple','avocado','banana','beer','biscuits','boisson-au-glucose-50g','bread-french-white-flour','bread-sourdough',
            'bread-white','bread-whole-wheat','bread-wholemeal','broccoli','butter','carrot','cheese','chicken','chips-french-fries','coffee-with-caffeine','corn','croissant','cucumber','dark-chocolate',
            'egg','espresso-with-caffeine','french-beans','gruyere','ham-raw','hard-cheese','honey','jam','leaf-spinach','mandarine','mayonnaise','mixed-nuts',
            'mixed-salad-chopped-without-sauce','mixed-vegetables','onion','parmesan','pasta-spaghetti','pickle','pizza-margherita-baked','potatoes-steamed','rice',
            'salad-leaf-salad-green','salami','salmon','sauce-savoury','soft-cheese','strawberries','sweet-pepper','tea','tea-green','tomato','tomato-sauce',
            'water','water-mineral','white-coffee-with-caffeine','wine-red','wine-white','zucchini']
    print('Prediction is a {}'.format(clas[idx]))
    print('Kalori :', df_kalori['kalori'][idx])