#include "modelisation/OpenGL.h"
#include "analyse/EdgeDetection.h"
#include "modelisation/Transformation.h"
#include "physics/AngleModel.h"
#include "physics/CollisionDetection.h"
#include "windows.h"
#include <string.h>
#include <iostream>
#include <vector>
#include <locale>
#include <sstream>
#include <conio.h>
#include <stdio.h>
#include <chrono>
#include <opencv2/highgui/highgui_c.h>
#include <WS2tcpip.h>

#include <time.h>

#include "database/Bdd.h"
#include <mysql.h>
#include "writing/CSVWriter.h"

#pragma comment(lib,"ws2_32.lib")

using namespace cv;
using namespace std;



CameraStream *cameraStream = nullptr;       //flux video
OpenGL *window = nullptr;                   //la fenetre OpenGL
AngleModel *angleModel = nullptr;           //sert a definir l angle visualise ?
Ball *ball = nullptr;                       //definit les parametres de la balle
__int64 diff = -1;                          //difference de temps entre la réception du signal sur l'android et la reception de la donnée sur le labyrinthe

/// Pour récupérer les données de stress
HANDLE myHandle = nullptr;                  
/*declaration de HANDLE ? pas compris
#ifdef STRICT
typedef void *HANDLE;
#if 0 && (_MSC_VER > 1000)
#define DECLARE_HANDLE(name) struct name##__; typedef struct name##__ *name
#else
#define DECLARE_HANDLE(name) struct name##__{int unused;}; typedef struct name##__ *name
#endif
#else
typedef PVOID HANDLE;
#define DECLARE_HANDLE(name) typedef HANDLE name
#endif
typedef HANDLE *PHANDLE;

*/
unsigned int progress = 0;                      //valeur actuelle du stress ?
int gameEnded = 0;
/// Pour afficher les FPS
int frame = 0, myTime, timebase = 0;            //frame= nb d image, mytime et timebase servent a compter une seconde
double fps = 0.0;                               //nombre d image par seconde

/// Prototypes des fonctions de ce fichier
void loop(int);                                 //declaration de la méthode loop
void setupMaze();                               //declaration de la méthode setupMaze
DWORD WINAPI getMuseResult(LPVOID lpProgress);  //declaration de la méthode getMuseResult
int numUser;
//string myT; 
int main(int argc, char** argv) {
    numUser = getnextID();
    int64 tick = 0;                             //recupere le nb de ms depuis le lancement de l appli
    bool timer = false;                         // ??? rien compris, la condition est trop bizarre
    bool anaglyph;                              //mode normal ou 3D
    string name = "aMAZEd Calibration";         //nom de la fenetre
    ball = new Ball(0.5, 0.5, 0.02, 50);        //cree un boule en donnant en parametre position en x, y, rayon, (et m?)
    cameraStream = new CameraStream();          //flux de la cam
    namedWindow(name, WINDOW_OPENGL);           //cree une fenetre supportant openGL avec le nom precise au dessus
    HWND* hwnd;                                 //???

    DWORD myThreadID;                           //???doit contenir un id de thread
    //myHandle = CreateThread(0, 0, getMuseResult,  &progress, 0, &myThreadID);//cree
    

    //Initialisation, montre l'image dont le chemin relatif est defini ci-dessous
    Mat img = imread("assets/init.png", IMREAD_COLOR);
    imshow("Initialisation", img);

    //en fonction de si on appuie sur espace (key==32), entree(key==13), ou une autre touche, on va passer respectivement en mode normal, mode 3D, ou sortir de l'application 
    int key = waitKey();
    if (key == 32) {
        anaglyph = false;
    }
    else if (key == 13) {
        anaglyph = true;
        //myHandle = CreateThread(0, 0, getMuseResult, &progress, 0, &myThreadID);//cree
    }
    else {
        return 0;
    }
    cvDestroyWindow("Initialisation");


    while (true) {
        Mat currentFrame = cameraStream->getCurrentFrame();                                         //recupere la matrice de l image renvoyee par la cam
        Size textSize1 = getTextSize("Normal Mode", FONT_HERSHEY_PLAIN, 4, 1, 0);                   //recupere la taille de Normal Mode
        Size textSize2 = getTextSize("3D Mode", FONT_HERSHEY_PLAIN, 4, 1, 0);                       //recupere la taille de 3D Mode
        Size textSize3 = getTextSize("3", FONT_HERSHEY_PLAIN, 6, 1, 0);                             //recupere la taille de 3
        Size textSize4 = getTextSize("Inclinaison trop horizontale", FONT_HERSHEY_PLAIN, 2, 1, 0);  //recupere la taille de Inclinaison trop horizontale
        Size textSize5 = getTextSize("Inclinaison trop verticale", FONT_HERSHEY_PLAIN, 2, 1, 0);    //recupere la taille de Inclinaison trop verticale
        if (!anaglyph) {//si on est en mode normal, mettre le texte "Normal Mode" avec les parametres definis
            putText(currentFrame, "Normal Mode", Point2i((currentFrame.cols - textSize1.width) / 2, currentFrame.rows - 30), FONT_HERSHEY_PLAIN, 4, Scalar(0, 0, 255), 2);
        }
        else if (anaglyph) {//si on est en mode 3D, mettre le texte "3D Mode" avec les parametres definis
            putText(currentFrame, "3D Mode", Point2i((currentFrame.cols - textSize2.width) / 2, currentFrame.rows - 30), FONT_HERSHEY_PLAIN, 4, Scalar(0, 0, 255), 2);
            //myHandle = CreateThread(0, 0, getMuseResult, &progress, 0, &myThreadID);//cree
        }
        float long12, long03, long01, long23, ratio = 0;
        EdgeDetection ED = EdgeDetection(currentFrame, true);       //detecte les bords du labyrinthe
        vector<Point2i> coordCorner = ED.getCorner(currentFrame);   //recupere les angles du labyrinthe (ca doit forcement etre un carre/rectangle, pas n'importe quel quadrilatere)

        /*
        * coordCorner[0] en haut a gauche  (normalement)
        * coordCorner[1] en haut a droite  (normalement)
        * coordCorner[2] en bas a droite (normalement)
        * coordCorner[3] en bas a gauche  (normalement)
        */  
        /*schéma
        *     x           x
        *   y 0           1 y
        * 
        * 
        *   y 3           2 y
        *     x           x
        */ 
        //test
        //racine de (difference de x des deux points de droite, le tout au carre, puis pareil pour les y, et enfin on additionne les 2 resultats)
        //ca nous donne tout simplement l hypotenuse du triangle rectangle forme par les 2 points sur un graphique (pour nico cest clair, lui demander si besoin)
        long12 = sqrt(pow(coordCorner[1].x - coordCorner[2].x, 2) + pow(coordCorner[1].y - coordCorner[2].y, 2));

        //racine de (difference de x des deux points de gauche, le tout au carre, puis pareil pour les y, et enfin on additionne les 2 resultats)
        //ca nous donne tout simplement l hypotenuse du triangle rectangle forme par les 2 points sur un graphique (pour nico cest clair, lui demander si besoin)
        long03 = sqrt(pow(coordCorner[0].x - coordCorner[3].x, 2) + pow(coordCorner[0].y - coordCorner[3].y, 2));

        //dans le cas ou un des 2 hypotenuse est significativement plus long que lautre, on va calculer la distance entre les 2 points du haut (0 et 1), et du bas (2 et 3), puis on va calculer le ratio du segment du haut sur le segment du bas
        if (long03 > long12 * 0.9 && long03 < long12 * 1.1) {
            long01 = sqrt(pow(coordCorner[0].x - coordCorner[1].x, 2) + pow(coordCorner[0].y - coordCorner[1].y, 2));
            long23 = sqrt(pow(coordCorner[2].x - coordCorner[3].x, 2) + pow(coordCorner[2].y - coordCorner[3].y, 2));
            ratio = long01 / long23;
        }
        //on considere que la feuille est correctement inclinee ssi le ratio est compris entre 0.73 et 0.8
        bool is45 = ratio > 0.73 && ratio < 0.8;

        if (ratio > 0.73 && ratio < 0.8) {

        }
        else if (ratio < 0.8) {
            putText(currentFrame, "Inclinaison trop horizontale", Point2i((currentFrame.cols - textSize4.width) / 2, currentFrame.rows - 100), FONT_HERSHEY_PLAIN, 2, Scalar(0, 0, 255), 2);
        }
        else if (ratio > 0.73) {
            putText(currentFrame, "Inclinaison trop verticale", Point2i((currentFrame.cols - textSize5.width) / 2, currentFrame.rows - 100), FONT_HERSHEY_PLAIN, 2, Scalar(0, 0, 255), 2);
        }


        //si on est considere a 45 degres et que le timer etait a faux jusque la, on le passe a vrai
        if (is45 && timer == false) {
            //tick = getTickCount();
            bool timer = true;
        }
        //sinon, tant que nous ne sommes pas bien positionnes, on va demander a l utilisateur de positionner le labyrinthe devant la camera
        else {
            tick = getTickCount();
            bool timer = false;
            putText(currentFrame, "Placez labyrinthe face a la camera", Point2i(0, 50), FONT_HERSHEY_PLAIN, 2, Scalar(0, 0, 255), 2);
        }
        //on va faire un compte a rebours tant que la feuille est bien inclinee, et au bout de 4 secondes si la feuille n a pas bouge, on va lancer le jeu
        int64 tick2 = getTickCount();
        int64 sec = (double)(tick2 - tick) / getTickFrequency();
        //cout << ratio << endl;

        if (sec == 1) {
            putText(currentFrame, "3", Point2i((currentFrame.cols - textSize3.width) / 2, 100), FONT_HERSHEY_PLAIN, 6, Scalar(0, 255, 0), 2);
        }
        if (sec == 2) {
            putText(currentFrame, "2", Point2i((currentFrame.cols - textSize3.width) / 2, 100), FONT_HERSHEY_PLAIN, 6, Scalar(0, 255, 0), 2);
        }
        if (sec == 3) {
            putText(currentFrame, "1", Point2i((currentFrame.cols - textSize3.width) / 2, 100), FONT_HERSHEY_PLAIN, 6, Scalar(0, 255, 0), 2);
        }
        if (sec == 4) {
            destroyWindow(name);//detruit la petite fenetre precedente
            //Lancement du jeu
            //myHandle = CreateThread(0, 0, getMuseResult, &progress, 0, &myThreadID);//cree
            Mat currentFrame = cameraStream->getCurrentFrame();//on capture l image et on la met dans une matrice
            double ratio = (double)currentFrame.cols / (double)currentFrame.rows; //definit le ratio entre les lignes et les colonnes capturees (pour nico : 640 par 480)
            int width = 1000; /// Largeur de la fenêtre
            auto* glutMaster = new GlutMaster();// ??? cree des "dummy variables (je vois pas l interet du coup et initialize glut avec la ligne "glutInit(&dummy_argc, dummy_argv);"
            window = new OpenGL(glutMaster, width, (int)(width / ratio), 0, 0, (char*)("aMAZEd"), ball, cameraStream, anaglyph, progress);//cree une nouvelle fenetre supportee par openGL
            //if (key==13) {
            myHandle = CreateThread(0, 0, getMuseResult, &progress, 0, &myThreadID);//cree
            //}
            setupMaze();// definit le labyrinthe
            window->startTimer();//commence a compter le temps que met l utilisateur a resoudre le labyrinthe

            glutMaster->CallGlutMainLoop(); // ??? appelle la fct de glut "glutMainLoop();"
        }


        hwnd = (HWND*)(cvGetWindowHandle(name.c_str()));
        if (hwnd == nullptr) {
            // CloseHandle(myHandle);
            delete cameraStream;
            delete window;
            delete angleModel;
            return 0;
        }
        imshow(name, currentFrame);

        int key = waitKey(20);

    }

    // CloseHandle(myHandle);
    delete cameraStream;
    delete window;
    delete angleModel;


    return 0;
}

//Fonction de calcul de distance entre deux points p1 et p2
double calculateDistance(cv::Point p1, cv::Point p2){
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

void loop(int endGame){

    window->setProgress(progress);

    if(endGame == 1){
        //CloseHandle(myHandle);
        waitKey(0);                 //on attend une touche du clavier (c'est ce que veut dire la valeur 0)
        return;
    }
    //myT= to_string((int)difftime(time(nullptr), start));
    /// Affichage FPS
    frame++;
    myTime = glutGet(GLUT_ELAPSED_TIME);
    if (myTime - timebase > 1000) {
        fps = frame * 1000.0 / (myTime - timebase);         //comme on est en milliseconde et pas en seconde, on est oblige de multiplier frame par 1000
        timebase = myTime;                                  //on redefini le compteur le plus eleve comme étant le plus bas pour la prochaine seconde
        frame = 0;                                          //et on reinitialise le nombre de frame compte
    }
    window->setFps(fps);                                    //definit le nombre de fps

    vector<Point2i> coordCorner;                            //les 4 angles du rectangle avec pour chacun les positions en x et y
    Mat currentFrame = cameraStream->getCurrentFrame();     //recuperer dans une matrice l image retournee par la cam
    EdgeDetection edgeDetection = EdgeDetection(currentFrame, false);   //detecter les cotes (et les murs ?)
    coordCorner = edgeDetection.getCorner(currentFrame);    //detecter les angles

    double offSetBall = 0.08;                               //pour rebondir en cas de collision (bizarre)

    /// Si les 4 coins ont été détéctées et que la distance absolue entre les x des 2 points du haut (0 et 1) est supérieure (!inferireure) à celle des 2 points du bas (2 et 3) ???condition etragne
    if (coordCorner.size() == 4 && !edgeDetection.isReversed(coordCorner)) {

        double aire = calculateDistance(coordCorner[0], coordCorner[1]) * calculateDistance(coordCorner[1], coordCorner[2]);
        //cout << aire << endl;

        /// Calcul de l'aire pour réduire les saccades
        if (aire > 20000.0 && aire < 170000.0) {

            Transformation transformation = Transformation(coordCorner, Size(currentFrame.cols, currentFrame.rows), 0.1, 20);
            angleModel->setCurrentTransformation(&transformation);

            vector<Wall> walls;
            if (CollisionDetection::findCollisions(ball, window->getWalls(), walls)) {

                //Incrémentation du nombre de collisions
                window->incrementNBCollisions(numUser);

                /// Detection de la nature de la collision
                bool verticalCollision = false;
                bool horizontalCollision = false;
                for (auto& wall : walls) {
                    if (!verticalCollision && wall.isVertical()) verticalCollision = true;
                    if (!horizontalCollision && !wall.isVertical()) horizontalCollision = true;
                }

                /// Collision verticale on rebondit selon l'axe X
                if (verticalCollision) {
                    ball->setNextX(ball->getNextX() - 1.5*ball->getVx());
                    //ball->setVx(-ball->getVx());
                    if (ball->getVx() >= 0) {
                        ball->setVx(-offSetBall);
                    }
                    else {
                        ball->setVx(offSetBall);
                    }
                    ball->setAx(0);
                }

                /// Collision horizontale on rebondit selon l'axe Y
                if (horizontalCollision) {
                    ball->setNextY(ball->getNextY() + 1.5*ball->getVy());
                    //ball->setVy(-ball->getVy());

                    if (ball->getVy() >= 0) {
                        ball->setVy(-offSetBall);
                    }
                    else {
                        ball->setVy(offSetBall);
                    }
                    ball->setAy(0);
                }

                ball->updatePosition();

                /// S'il s'agit d'une collision sur le bout du mur
                /*if(CollisionDetection::findCollisions(ball, window->getWalls(), walls)){
                    if(verticalCollision){
                        ball->setNextY(ball->getNextY() - ball->getVy() * 2);
                        if(ball->getVy() > 0){
                            ball->setVy(-offSetBall);
                        }else{
                            ball->setVy(offSetBall);
                        }
                        ball->setAy(0);
                    }

                    if(horizontalCollision){
                        ball->setNextX(ball->getNextX() - ball->getVx() * 2);
                        if(ball->getVx() > 0){
                            ball->setVx(-offSetBall);
                        }else{
                            ball->setVx(offSetBall);
                        }
                        ball->setAx(0);
                    }
                }*/

                //cout << "R=" << ball->getR() << endl;


            }
            else {
                ball->setAx(angleModel->getAngleY() / 10);
                ball->setAy(angleModel->getAngleX() / 10);
                ball->updatePosition();
            }

            double p[16];
            double m[16];
            transformation.getProjectionMatrix(p);
            transformation.getModelviewMatrix(m);
            window->setProjectionMatrix(p);
            window->setModelviewMatrix(m);
        }
    }

    glutPostRedisplay();

}

void setupMaze(){

    /// Détection des coins
    Mat currentFrame = cameraStream->getCurrentFrame();
    EdgeDetection edgeDetection = EdgeDetection(currentFrame, true);

    vector<Point2i> coordCorner;
    vector<Point2i> coordStartEnd;
    vector<vector<Point2i>> lines;

    /// Tant que les 4 coins n'ont pas été détéctées
    do {

        currentFrame = cameraStream->getCurrentFrame();
        coordStartEnd = edgeDetection.startEndDetection(currentFrame);
        coordCorner = edgeDetection.getCorner(currentFrame);

        /// Detection des murs
        lines = edgeDetection.wallsDetection(currentFrame, coordCorner, coordStartEnd);

    }while(coordStartEnd.size() != 2);

    Transformation *transformation = new Transformation(coordCorner, Size(currentFrame.cols, currentFrame.rows), 1, 10);

    ///point d'arrivée sauvegarde
    Point2d *pointModelEnd = new Point2d(transformation->getModelPointFromImagePoint(coordStartEnd[1]));
    window->setEndPoint(pointModelEnd);

    ///set la boule aux coordonnées du départ détectés
    cv::Point2d pointModelStart = transformation->getModelPointFromImagePoint(coordStartEnd[0]);

    ///set la boule aux coordonnées du départ
    ball->setNextX(pointModelStart.x);
    ball->setNextY(pointModelStart.y);

    /// Calcul des coordonées des extrimités des murs
    vector<Wall> walls;
    for (const auto &line : lines) {

        Point2d pointImageA = transformation->getModelPointFromImagePoint(line[0]);
        Point2d pointImageB = transformation->getModelPointFromImagePoint(line[1]);

        Wall wall(pointImageA, pointImageB);

        walls.push_back(wall);
    }

    /// Murs extérieurs
//    walls.emplace_back(Point2d(0, 0), Point2d(0, 1));
//    walls.emplace_back(Point2d(1, 1), Point2d(0, 1));
//    walls.emplace_back(Point2d(1, 1), Point2d(1, 0));
//    walls.emplace_back(Point2d(1, 0), Point2d(0, 0));

    window->setWalls(walls);

    angleModel = new AngleModel(transformation);


}


DWORD WINAPI getMuseResult(LPVOID lpProgress) {

    unsigned int& progress = *((unsigned int*)lpProgress);

    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        cout << "WSAStartup failed.\n";
        system("pause");
        return 1;
    }


    // Create a server hint structure for the server
    sockaddr_in serverHint;
    //serverHint.sin_addr.S_un.S_addr = ADDR_ANY; // Us any IP address available on the machine
    //serverHint.sin_addr.S_un.S_addr = ADDR_ANY; // Us any IP address available on the machine
    //serverHint.sin_addr.S_un.S_addr = "10.112.63.8"; // Us any IP address available on the machine
    //serverHint.sin_addr.S_un.S_addr = INADDR_ANY; // Us any IP address available on the machine
    //serverHint.sin_addr.s_addr = INADDR_ANY; // Us any IP address available on the machine
    serverHint.sin_addr.s_addr = htonl(ADDR_ANY);
    serverHint.sin_family = AF_INET; // Address format is IPv4
    //serverHint.sin_port = htons(9436); // Convert from little to big endian
    serverHint.sin_port = htons(9436); // Convert from little to big endian

    SOCKET in = socket(AF_INET, SOCK_DGRAM, 0);
    printf("%d", in);
    //SOCKET in = socket(AF_INET, SOCK_STREAM, 0);
    if (in ==-1){
        printf("ERROR creating socket");
    }

    // Try and bind the socket to the IP and port
    int iResult= ::bind(in, (sockaddr*)&serverHint, sizeof(serverHint));
    
    if (iResult == SOCKET_ERROR)
    {
        cout << "Can't bind socket! " << WSAGetLastError() << endl;
        //closesocket(in);
        return 1;
    }

    ////////////////////////////////////////////////////////////
    // MAIN LOOP SETUP AND ENTRY
    ////////////////////////////////////////////////////////////

    sockaddr_in client; // Use to hold the client information (port / ip address)
    int clientLength = sizeof(client); // The size of the client information

    char buffer[1024];

    while(true){
        
        ZeroMemory(&client, clientLength); // Clear the client structure
        ZeroMemory(buffer, 1024); // Clear the receive buffer

        // Wait for message
        //printf("in:%d, buffer[0]: %d, client: %d, clientLength: %d", in, buffer[0], &client, &clientLength);

        /*
        recvfrom(
        _In_ SOCKET s,
        _Out_writes_bytes_to_(len, return) __out_data_source(NETWORK) char FAR * buf,
        _In_ int len,
        _In_ int flags,
        _Out_writes_bytes_to_opt_(*fromlen, *fromlen) struct sockaddr FAR * from,
        _Inout_opt_ int FAR * fromlen
        );
        */
        int bytesIn = recvfrom(in, buffer, 1024, 0, (sockaddr*)&client, &clientLength);
        //int bytesIn = recvfrom(in, buffer, 1024, 0, (struct sockaddr*)&client, &clientLength);
        if (bytesIn == SOCKET_ERROR)
        {
            cout << "Error receiving from client " << WSAGetLastError() << endl;
            continue;
        }

        // Display message and client info
        char clientIp[256]; // Create enough space to convert the address byte array
        ZeroMemory(clientIp, 256); // to string of characters

        // Convert from byte array to chars
        inet_ntop(AF_INET, &client.sin_addr, clientIp, 256);

        // Display the message / who sent it
        cout << buffer << endl;

        int start = 0;
        while (buffer[start] != '=') start++;
        int middle = start + 1;
        while (buffer[middle] != '/') middle++;
        int end = start + 1;
        while (buffer[end] != ';') end++;

        //Isolation des données de stress
        char firstProgress = buffer[start];
        char lastProgress = buffer[start+1];

        string progress_str;
        for (int i = start+1; i < middle; i++) {
            progress_str.push_back(buffer[i]);
        }

        progress = (unsigned int)atoi(progress_str.c_str());

        //Isolation des données de date
        string time_str;
        for (int i = start + 2; i < end; i++) {
            time_str.push_back(buffer[i]);
        }

        __int64 t = _atoi64(time_str.c_str());
        __int64 now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        
        //différence de temps entre la réception du signal sur l'android et la réception de la donnée sur le labyrinthe
        if(diff < 0){
            diff = now - t;
        } else {
            diff = (diff + now - t)/2;
        }
        if (gameEnded != 1){
            CSVWriter writeObject("statTempsReel.csv", numUser, (int)difftime(time(nullptr), window->getStart()), NULL, progress);
            cout << (int)difftime(time(nullptr), window->getStart()) << endl;
            if (!writeObject.doesFileExists()) {
                //Création des colonnes si le fichier n'existe pas
                writeObject.writeColumnsName();
                cout << "oskour" << endl;
            }
            //Ecriture d'une nouvelle ligne
            writeObject.writeLine();
            //cout << "Diff:" << diff << endl;

            //printf("progress:%d, bytesIn:%d, time:%d, now:%d", progress, bytesIn, (int)time, (int)now);
        }
    }
    cout<<"fin du while true"<<endl;
}
