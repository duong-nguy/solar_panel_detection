from preprocess import FeatureEngineer,Preprocessor,PerturbationRank,FeatureSelection
from train import F1Score,unet,unet_plus_2d,Train
from predict import Predict



def main():
    modules = []
    project_root = 'spd'
    train_path = 'data/train/s2_image'
    target_path = 'data/train/mask'
    test_path =  'data/eval'
    sample_path = 'data/eval/samples'
    preprocess_output = (26,26,20)
    train_input= (32,32,12)
    epochs = 500 
    batch = 129
    learning_rate = 3e-6
    
    modules.append(
            FeatureEngineer(project_root=project_root,train_path=train_path,
                            target_path=target_path,test_path=test_path,
                            output_shape=preprocess_output))
    modules.append(
            Preprocessor(project_root=project_root,train_path=train_path,
                         target_path=target_path,test_path=test_path,
                         output_shape=preprocess_output))
    modules.append(
            PerturbationRank(project_root=project_root,hypermodel=unet_plus_2d,
                            batch=8,epochs=80))
    modules.append(
            FeatureSelection(
                    threshold=0.3,project_root=project_root,
                    train_path=train_path,target_path=target_path,
                    test_path=test_path,output_shape=train_input))
    modules.append(
            Train(
                project_root=project_root,model=unet_plus_2d,# unet,unet_plus_2d
                loss=hybrid_loss,#'binary_crossentropy'
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                batch=batch,epochs=epochs,metrics=F1Score()))

    modules.append(
            Predict(project_root,test_path=test_path,sample_path=sample_path))

    for module in modules:
        module.run()

if __name__ == '__main__':
    main()


