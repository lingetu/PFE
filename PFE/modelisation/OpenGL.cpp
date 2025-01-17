#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "OpenGL.h"
#include "../physics/CollisionDetection.h"
#include "../writing/CSVWriter.h"
#include <ctime>
#include <iomanip>
#include <sstream>

#include "../database/Bdd.h"
#include <mysql.h>

using namespace cv;
using namespace std;
int level = 1;
// Fonction appelé en boucle et définie dans le main
void loop(int);

OpenGL::OpenGL(GlutMaster * glutMaster, int setWidth, int setHeight, int setInitPositionX, int setInitPositionY, char * title, Ball *ball, CameraStream * cameraStream, bool anaglyph, unsigned int progress){

    this->ball = ball;
    this->cameraStream = cameraStream;

    ///Muse
    this->progress = progress;

    this->width  = setWidth;
    this->height = setHeight;

    this->initPositionX = setInitPositionX;
    this->initPositionY = setInitPositionY;

    this->p = new double[16];
    this->m = new double[16];

    this->anaglyph = anaglyph;

    this->nbCollisions = 0;
    this->isWritten = false;

    this->textMaze = cv::imread("assets/gazon1.jpg"); //texture du sol du labyrinthe
    this->textWall = cv::imread("assets/mazeWall.png"); //texture du mur du labyrinthe
    this->textFlag = cv::imread("assets/mazeFlag.png"); //texture du drapeau d'arrivée
    this->cap = VideoCapture("assets/video.mp4");
    
     if(anaglyph){
        this->textCam = imread("assets/noir.jpg"); // fond noir
    }

    glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE | GLUT_STENCIL);
    glutInitWindowSize(this->width, this->height);
    glutInitWindowPosition(this->initPositionX, this->initPositionY);
    glViewport(0, 0, this->width, this->height);

    glutMaster->CallGlutCreateWindow(title, this);
    glGenTextures(4, textArray);
    loadTexture(textArray[ID_TEXT_MAZE], textMaze);
    loadTexture(textArray[ID_TEXT_WALL], textWall);
    loadTexture(textArray[ID_TEXT_FLAG], textFlag);
    applicateLight();
    glEnable(GL_DEPTH_TEST);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1); 	// Nécessaire pour éviter une déformation de l'image
}

OpenGL::~OpenGL(){
    glutDestroyWindow(windowID);
    delete p;
    delete m;
}

void OpenGL::CallBackDisplayFunc(){

    bool endGame = false;


    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glEnable(GL_TEXTURE_2D);
    
      if(!click && !anaglyph){
          bool ret;
          ret = this->cap.read(this->textCam);
          if (ret == false) {
              this->cap.set(CAP_PROP_POS_FRAMES,0);
              this->cap.read(this->textCam);
          }
    }

    if(!anaglyph && click){
        this->textCam = cameraStream->getCurrentFrame();
    }

    float startXProgress = 20;
    float startYProgress = 20;
    float widthProgress  = 200;
    float heighProgress = 30;

    Scalar rouge = Scalar(0, 0, 255);
    Scalar vert = Scalar(0, 255, 0);

    //Ajout de la donnée de stress au vecteur 
    progressData.push_back(progress);
    if(progress == 0){
    }
    else if(progress >= 50){

        double alphaMax = 0.50;
        double alphaMin = 0;

        double valMax = 100;
        double valMin = 50;


        //APPLY A FILTER
        cv::Mat filtre = this->textCam(cv::Rect(0, 0, this->textCam.cols, this->textCam.rows));
        cv::Mat color(filtre.size(), CV_8UC3, vert);
        int val = (int)valMax - this->progress;
        double alpha = 1 - ((alphaMax - alphaMin)/(valMax - valMin)) * val + alphaMin;
        cv::addWeighted(color, alpha, filtre, 1.0 - alpha , 0.0, filtre);

    }
    else{

        double alphaMax = 0.40;
        double alphaMin = 0;

        double valMax = 49;
        double valMin = 0;


        //APPLY A FILTER
        cv::Mat filtre = this->textCam(cv::Rect(0, 0, this->textCam.cols, this->textCam.rows));
        cv::Mat color(filtre.size(), CV_8UC3, rouge);
        int val = (int)valMax - this->progress;
        double alpha = ((alphaMax - alphaMin)/(valMax - valMin)) * val + alphaMin;
        cv::addWeighted(color, alpha, filtre, 1.0 - alpha , 0.0, filtre);
    }

    //Jauge de stress
    rectangle(this->textCam ,
        Point2f (startXProgress ,  startYProgress),        
        Point2f (startXProgress + ((float)((this->progress)*widthProgress))/100 , startYProgress + heighProgress ),
        vert,
        - 1 ,
        8);

    rectangle(this->textCam ,
        Point2f (startXProgress + ((float)((this->progress)*widthProgress))/100,  startYProgress), 
        Point2f (startXProgress + widthProgress , startYProgress + heighProgress),
        rouge,
        - 1 ,
        8);

    //string pressToExit = "press q to exit";
    putText(this->textCam, to_string(fps), Point2i(0, 10), FONT_HERSHEY_PLAIN, 0.9, Scalar(0, 0, 255), 1);
    //putText(this->textCam, pressToExit, Point2i(this->textCam.cols, 10), FONT_HERSHEY_PLAIN, 0.6, Scalar(0, 0, 255), 1);

    //progress Muse (le 0)
    putText(this->textCam, to_string(this->progress), Point2i(this->textCam.cols - (int)this->textCam.cols / 2, 50), FONT_HERSHEY_PLAIN, 4, Scalar(255, 0, 0), 3);
    // temps en direct
    putText(this->textCam, "Temps : " + to_string((int)difftime(time(nullptr), start)), Point2i(0, 440), FONT_HERSHEY_PLAIN, 1.5, Scalar(255, 0, 0), 1.5);
    // Nombre de collisions
    putText(this->textCam, "Nombre de collisions : " + to_string(getNBCollisions()), Point2i(0, 470), FONT_HERSHEY_PLAIN, 1.5, Scalar(255, 0, 0), 1.5);

    // Si la balle atteint la fin on affiche l'écran de fin avec le temps
    if (CollisionDetection::hasArrived(ball, this->getEndPoint())) {
        destroyAllWindows();

        //Calcul du stress moyen
        double progressAverage = 0.0;
        for (int i = 0; i < progressData.size(); i++) {
            progressAverage = progressAverage + progressData.at(i);
        }
        progressAverage = (progressAverage / progressData.size());

        std::ostringstream streamAverage;
        // Set Fixed -Point Notation
        streamAverage << std::fixed << setprecision(2) << progressAverage;
        // Get string from output string stream
        std::string stringAverage = streamAverage.str();

        //Taille du texte pour centrer les données
        /*Size textSizeTemps = getTextSize(to_string((int)difftime(time(nullptr), start)) + "s", FONT_HERSHEY_PLAIN, 4, 4, 0);
        Size textSizeCol = getTextSize(to_string(getNBCollisions()), FONT_HERSHEY_PLAIN, 4, 4, 0);
        Size textSizeAverage = getTextSize(to_string(progressAverage), FONT_HERSHEY_PLAIN, 4, 4, 0);*/

        //Affichage du texte
        this->textCam = imread("assets/mazeEnd2.png");
        putText(this->textCam, to_string((int)difftime(time(nullptr), start)) + "s", Point2i(635, 330), FONT_HERSHEY_PLAIN, 4, Scalar(225, 238, 251), 4);
        putText(this->textCam, to_string(getNBCollisions()), Point2i(635, 445), FONT_HERSHEY_PLAIN, 4, Scalar(225, 238, 251), 4);
        putText(this->textCam, stringAverage + "%", Point2i(635, 580), FONT_HERSHEY_PLAIN, 4, Scalar(225, 238, 251), 4);

        endGame = true;


        if (!this->isWritten) {
            //Bdd* maDB = nullptr;
            //Bdd *maDB = new Bdd();
            //maDB->dbCommunication();
            //maDB->dbCommunication();
            dbCommunication(getNBCollisions(), progressAverage, level, (int)difftime(time(nullptr), start));
            //(int)difftime(time(nullptr), start), getNBCollisions(), progressAverage
            //(int nbCollisions, float stress, int level, int gameTime)
            /////Sauvegarde des données dans un fichier
            ///CSVWriter writeObject("statistiques.csv", (int)difftime(time(nullptr), start), getNBCollisions(), progressAverage);
            ///
            /////Vérification de l'existence du fichier
            ///if (!writeObject.doesFileExists()) {
            ///    //Création des colonnes si le fichier n'existe pas
            ///    writeObject.writeColumnsName();
            ///}
            /////Ecriture d'une nouvelle ligne
            ///writeObject.writeLine();
            this->isWritten = true;
        }

        

        cout << "Bonsoir" << endl;
    }


    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, -10, 10);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    drawBackground();


    if(!endGame) {

        glDisable(GL_CULL_FACE);
        glMatrixMode(GL_PROJECTION);
        glLoadMatrixd(this->p);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        if(anaglyph){
            double offset;
            offset = 0.01;

            double modelviewTemp[16];

            offsetCamera(offset, modelviewTemp);
            glLoadMatrixd(modelviewTemp);
            filtreRouge();
            drawElements();

            offsetCamera(-offset, modelviewTemp);
            glLoadMatrixd(modelviewTemp);
            filtreBleu();
            drawElements();

        }else{
            glLoadMatrixd(this->m);
            drawElements();
        }

    }

    glutSwapBuffers();

    /// Si la partie est terminée renvoie 1 pour signaler l'arret du programme
    if(endGame){
        glutTimerFunc((unsigned int)1000 / MAX_FPS, loop, 1);
    }else{
        glutTimerFunc((unsigned int)1000 / MAX_FPS, loop, 0);
    }
}

int OpenGL::getProgress() const {
    return progress;
}

void OpenGL::setProgress(int progress) {
    OpenGL::progress = progress;
}

time_t OpenGL::getStart()
{
    return start;
}

void OpenGL::CallBackReshapeFunc(int w, int h){

    this->width = w;
    this->height= h;

    glViewport(0, 0, this->width, this->height);
    CallBackDisplayFunc();
}

void OpenGL::CallBackKeyboardFunc(unsigned char key, int x, int y) {
    if(key == 'q'){
        exit(0);
    }
   if(key == 'f')
           if (click == true)  click = false;
           else  click = true;
}

void OpenGL::CallBackIdleFunc(){
    CallBackDisplayFunc();
}

void OpenGL::loadTexture(GLuint id, Mat img) {

    glBindTexture(GL_TEXTURE_2D, id);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

    glTexImage2D(GL_TEXTURE_2D,     /// Type of texture
                 0,                 /// Pyramid level (for mip-mapping) - 0 is the top level
                 GL_RGB,            /// Internal colour format to convert to
                 img.cols,          /// Image width  i.e. 640 for Kinect in standard mode
                 img.rows,          /// Image height i.e. 480 for Kinect in standard mode
                 0,                 /// Border width in pixels (can either be 1 or 0)
                 0x80E0,            /// Valeur correspondante à GL_BGR
                 GL_UNSIGNED_BYTE,  /// Image data type
                 img.data);         /// The actual image data itself

}

void OpenGL::drawAxes(){

    glColor3f(1, 0, 0);
    glBegin(GL_LINE_STRIP);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(1.0f, 0.0f, 0.0f);
    glVertex3f(0.75f, 0.25f, 0.0f);
    glVertex3f(0.75f, -0.25f, 0.0f);
    glVertex3f(1.0f, 0.0f, 0.0f);
    glVertex3f(0.75f, 0.0f, 0.25f);
    glVertex3f(0.75f, 0.0f, -0.25f);
    glVertex3f(1.0f, 0.0f, 0.0f);
    glEnd();
    glColor3f(0, 1, 0);
    glBegin(GL_LINE_STRIP);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 1.0f, 0.0f);
    glVertex3f(0.0f, 0.75f, 0.25f);
    glVertex3f(0.0f, 0.75f, -0.25f);
    glVertex3f(0.0f, 1.0f, 0.0f);
    glVertex3f(0.25f, 0.75f, 0.0f);
    glVertex3f(-0.25f, 0.75f, 0.0f);
    glVertex3f(0.0f, 1.0f, 0.0f);
    glEnd();
    glColor3f(0, 0, 1);
    glBegin(GL_LINE_STRIP);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 1.0f);
    glVertex3f(0.25f, 0.0f, 0.75f);
    glVertex3f(-0.25f, 0.0f, 0.75f);
    glVertex3f(0.0f, 0.0f, 1.0f);
    glVertex3f(0.0f, 0.25f, 0.75f);
    glVertex3f(0.0f, -0.25f, 0.75f);
    glVertex3f(0.0f, 0.0f, 1.0f);
    glEnd();

    glColor4ub(255, 255, 0, 255);
    glRasterPos3f(1.1f, 0.0f, 0.0f);

    glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, 'x');
    glRasterPos3f(0.0f, 1.1f, 0.0f);
    glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, 'y');
    glRasterPos3f(0.0f, 0.0f, 1.1f);
    glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, 'z');

    glColor3f(1, 1, 1);

}

void OpenGL::displayText( float x, float y, int r, int g, int b, const char *string ) {
    size_t j = strlen(string);

    glColor3f( r, g, b );
    glRasterPos2f( x, y );
    for( int i = 0; i < j; i++ ) {
        glutBitmapCharacter( GLUT_BITMAP_TIMES_ROMAN_24, string[i] );
    }
}


void OpenGL::drawMazeGround(){
    glBindTexture(GL_TEXTURE_2D, textArray[ID_TEXT_MAZE]);

    /// Plateau de jeu
    glBegin(GL_POLYGON);
    glTexCoord2d(0, 1);glVertex3f(0.0f, 0.0f, 0.0f);
    glTexCoord2d(1, 1);glVertex3f(1.0f, 0.0f, 0.0f);
    glTexCoord2d(1, 0);glVertex3f(1.0f, 1.0f, 0.0f);
    glTexCoord2d(0, 0);glVertex3f(0.0f, 1.0f, 0.0f);
    glEnd();
}

void OpenGL::drawBackground() {
    loadTexture(textArray[ID_TEXT_CAM], this->textCam);
    glBegin(GL_POLYGON);
    glTexCoord2d(0, 1);glVertex3f(0.0, 0.0f, -5.0f);
    glTexCoord2d(0, 0);glVertex3f(0.0f, 1.0f, -5.0f);
    glTexCoord2d(1, 0);glVertex3f(1.0f, 1.0f, -5.0f);
    glTexCoord2d(1, 1);glVertex3f(1.0f, 0.0f, -5.0f);
    glEnd();
}

void OpenGL::drawElements() {

    if(!anaglyph) drawMazeGround();

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHT0);
    glEnable(GL_LIGHTING);
    drawWalls();
    drawFlag();
    glDisable(GL_TEXTURE_2D);
    applicateMaterial();
    glColor3f(1, 1, 1);
    ball->draw();


    GLfloat sol[3][3] = {{0.0f, 0.0f, 0.0f},
                         {1.0f, 0.0f, 0.0f},
                         {0.0f, 1.0f, 0.0f}};
    GLfloat lightPos[4] = {0.0f, 0.0f, 10.0f, 1.0};
    GLfloat ombre[4][4];

    // On utilise les faces avants seulement
    // ... il faudrait l'utiliser tout le temps ... mais il y a des polygones à l'envers :-( !! )
    // ... il y a un "soucis avec ça" ...

    // La matrice de transformation
    shadowMatrix(sol, lightPos, ombre);

    // Ecriture dans le stencil buffer
    // Pour écrire dans le stencil buffer, on utilise ni
    // le test de profondeur et on ne tient pas compte de la couleur
    glDisable(GL_DEPTH_TEST);
    glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
    // Tracé dans le stencil buffer (les points du sol à '1')
    glEnable(GL_STENCIL_TEST);
    glStencilOp(GL_REPLACE, GL_REPLACE, GL_REPLACE);
    glStencilFunc(GL_ALWAYS, 1, 0xffffffff); // c'est ce '1'
    drawMazeGround();

    // On a a nouveau besoin du tampon de profondeur et de la couleur
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glEnable(GL_DEPTH_TEST);

    // On va afficher seulement les valeurs '1' du stencil
    glStencilFunc(GL_EQUAL, 1, 0xffffffff); // c'est ce '1'
    glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);

    // Tracé
    // Pour la transparence de l'ombre
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    // Pour pouvoir utiliser glColor
    glDisable(GL_LIGHTING);
    // Ombre noire, "transparence" -> 0.5
    glColor4f(0.0f, 0.0f, 0.0f, 0.5f);

    glEnable(GL_CULL_FACE);
    glPushMatrix();
    glMultMatrixf((GLfloat *) ombre);
    ball->draw();
    glPopMatrix();

    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
    glDisable(GL_CULL_FACE);
    glEnable(GL_LIGHTING);
    glDisable(GL_BLEND);
    glDisable(GL_STENCIL_TEST);


}

void OpenGL::offsetCamera(double offset, double *mat) {
    for(int i = 0; i < 16; i++){
        mat[i] = this->m[i];
    }
    mat[12] += offset;

    if(offset > 0) mat[14] += offset;
}

void OpenGL::filtreRouge() {
    glColorMask(GL_TRUE, GL_FALSE, GL_FALSE, GL_TRUE);
}

void OpenGL::filtreBleu() {
    glColorMask(GL_FALSE, GL_FALSE, GL_TRUE, GL_TRUE);
}

void OpenGL::applicateMaterial() {
    GLfloat Lemission[4] = {0.023125f, 0.023125f, 0.023125f, 1.0f};
    GLfloat Ldiffuse[4] = {0.2775f, 0.2775f, 0.2775f, 1.0f};
    GLfloat Lspecular[4] = {0.773911f, 0.773911f, 0.773911f, 1.0f};
    GLfloat Lshininess[1] = {89.6f};
    glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, Lemission);
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, Ldiffuse);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, Lspecular);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, Lshininess);
}

void OpenGL::drawFlag(){

    glBindTexture(GL_TEXTURE_2D, textArray[ID_TEXT_FLAG]);

    ///poteau du drapeau
    glMatrixMode(GL_MODELVIEW);
    GLUquadric* params;
    params = gluNewQuadric();

    glPushMatrix();
    glTranslated(this->endPoint->x, this->endPoint->y, -FLAG_PIPE_HEIGHT);
    gluCylinder(params,FLAG_PIPE_RADIUS,FLAG_PIPE_RADIUS,FLAG_PIPE_HEIGHT,20,1); //(..., rayon bas, rayon haut, hauteur,  maillage, "stack")
    glPopMatrix();

    ///drap du drapeau
    glBegin(GL_POLYGON);
    glTexCoord2d(0, 1);
    glVertex3d(this->endPoint->x , this->endPoint->y - FLAG_PIPE_RADIUS, -FLAG_PIPE_HEIGHT);
    glTexCoord2d(1, 1);
    glVertex3d(this->endPoint->x , this->endPoint->y - FLAG_PIPE_RADIUS + FLAG_TOP_SIZE, -FLAG_PIPE_HEIGHT);
    glTexCoord2d(1, 0);
    glVertex3d(this->endPoint->x , this->endPoint->y - FLAG_PIPE_RADIUS + FLAG_TOP_SIZE, -FLAG_PIPE_HEIGHT - FLAG_TOP_SIZE);
    glTexCoord2d(0, 0);
    glVertex3d(this->endPoint->x , this->endPoint->y - FLAG_PIPE_RADIUS, -FLAG_PIPE_HEIGHT - FLAG_TOP_SIZE);
    glEnd();

}

void OpenGL::applicateLight() {
    //Light
    glEnable(GL_LIGHTING);
    GLfloat LPosition[4] =  { 1.5f, 1.5f, 0.0f, 1.0};
    GLfloat LAmbient[4] =  { 1.4, 1.4, 1.4, 1.0};
    GLfloat LDiffuse[4] =  {2.0, 2.0, 2.0, 1.0};
    GLfloat LSpecular[4] =  {1.0, 1.0, 1.0, 1.0};
    //GLfloat LPosition[4] = {1.5f, 1.5f, 0.0f, 1.0f};
    //GLfloat LAmbient[4]  = {0.0f, 0.0f, 0.5f, 0.5f };
    //GLfloat LDiffuse[4]  = {0.0f, 0.0f, 0.5f, 0.5f};
    //GLfloat LSpecular[4] = {1.0f, 1.0f, 1.0f, 0.5f};
    //float noAmbient[] = { 0.0f, 0.0f, 0.2f, 1.0f };       //low ambient light
    //float diffuse[] = { 0.0f, 0.0f, 1.0f, 1.0f };
    //float position[] = { 0.0f, 0.0f, 0.0f, 1.0f };
    glLightfv(GL_LIGHT0, GL_POSITION, LPosition);   // position
    glLightfv(GL_LIGHT0, GL_AMBIENT, LAmbient );    // couleur de la forme
    glLightfv(GL_LIGHT0, GL_DIFFUSE, LDiffuse);     // couleur de la lumière
    glLightfv(GL_LIGHT0, GL_SPECULAR, LSpecular);   // couleur du reflet
    glEnable(GL_LIGHT0);
}

void OpenGL::shadowMatrix(GLfloat points_plan[3][3], const GLfloat lightPos[4], GLfloat destMat[4][4]) {
    GLfloat planeCoeff[4];
    GLfloat dot;

    //on calcule un vecteur normal à ce plan
    normal(points_plan,planeCoeff);

    // le dernier coefficient se calcule par substitution
    planeCoeff[3] = - ( (planeCoeff[0]*points_plan[2][0]) + (planeCoeff[1]*points_plan[2][1]) + (planeCoeff[2]*points_plan[2][2]));
    dot = planeCoeff[0] * lightPos[0] + planeCoeff[1] * lightPos[1] + planeCoeff[2] * lightPos[2] + planeCoeff[3] * lightPos[3];

    // maintenant, on projette
    // 1ère colonne
    destMat[0][0] = dot - lightPos[0] * planeCoeff[0];
    destMat[1][0] = 0.0f - lightPos[0] * planeCoeff[1];
    destMat[2][0] = 0.0f - lightPos[0] * planeCoeff[2];
    destMat[3][0] = 0.0f - lightPos[0] * planeCoeff[3];

    // 2ème colonne
    destMat[0][1] = 0.0f - lightPos[1] * planeCoeff[0];
    destMat[1][1] = dot - lightPos[1] * planeCoeff[1];
    destMat[2][1] = 0.0f - lightPos[1] * planeCoeff[2];
    destMat[3][1] = 0.0f - lightPos[1] * planeCoeff[3];

    // 3ème colonne
    destMat[0][2] = 0.0f - lightPos[2] * planeCoeff[0];
    destMat[1][2] = 0.0f - lightPos[2] * planeCoeff[1];
    destMat[2][2] = dot - lightPos[2] * planeCoeff[2];
    destMat[3][2] = 0.0f - lightPos[2] * planeCoeff[3];

    // 4ème colonne
    destMat[0][3] = 0.0f - lightPos[3] * planeCoeff[0];
    destMat[1][3] = 0.0f - lightPos[3] * planeCoeff[1];
    destMat[2][3] = 0.0f - lightPos[3] * planeCoeff[2];
    destMat[3][3] = dot - lightPos[3] * planeCoeff[3];
}

void OpenGL::normal(float v[3][3], float out[3]) {

    float v1[3],v2[3];
    static const int x = 0;
    static const int y = 1;
    static const int z = 2;

    // Calcul de 2 vecteurs à partir des 3 points
    v1[x] = v[0][x] - v[1][x];
    v1[y] = v[0][y] - v[1][y];
    v1[z] = v[0][z] - v[1][z];
    v2[x] = v[1][x] - v[2][x];
    v2[y] = v[1][y] - v[2][y];
    v2[z] = v[1][z] - v[2][z];

    // calcul du produit vectoriel
    out[x] = (v1[y]*v2[z] - v1[z]*v2[y]);
    out[y] = (v1[z]*v2[x] - v1[x]*v2[z]);
    out[z] = (v1[x]*v2[y] - v1[y]*v2[x]);

    // on le réduit à un vecteur unité
    vecteurUnite(out);
}

void OpenGL::vecteurUnite(float vector[3]) {
    float length;

    // Calcul de la norme du vecteur
    length = sqrt((vector[0]*vector[0]) + (vector[1]*vector[1]) + (vector[2]*vector[2]));

    if(length == 0.0f) length = 1.0f;  //évite une violente erreur !!!
    vector[0] /= length;
    vector[1] /= length;
    vector[2] /= length;
}

void OpenGL::drawWalls() {

    glPushMatrix();
    glBindTexture(GL_TEXTURE_2D, textArray[ID_TEXT_WALL]);

    glEnable(GL_DEPTH_TEST);

    /// Pour chacune des lignes
    for(auto &wall : this->walls){
        wall.draw();
    }

    glPopMatrix();
}

const vector<Wall> &OpenGL::getWalls() const {
    return walls;
}

void OpenGL::setWalls(const std::vector<Wall> &walls) {
    this->walls = walls;
}

Point2d *OpenGL::getEndPoint(){
    return this->endPoint;
}

void OpenGL::setEndPoint(cv::Point2d *point){
    this->endPoint = point;
}

void OpenGL::setProjectionMatrix(const double *p) {
    for(int i = 0; i < 16; i++){
        this->p[i] = p[i];
    }
}

void OpenGL::setModelviewMatrix(const double *m) {
    for(int i = 0; i < 16; i++){
        this->m[i] = m[i];
    }
}

void OpenGL::setFps(double fps) {
    OpenGL::fps = fps;
}

void OpenGL::startTimer() {
    this->start = time(nullptr);
}

int OpenGL::getNBCollisions() {
    return this->nbCollisions;
}

void OpenGL::incrementNBCollisions(int numUser) {
    this->nbCollisions++;

    CSVWriter writeObject("statTempsReel.csv", numUser,(int)difftime(time(nullptr), start), nbCollisions, NULL);
    if (!writeObject.doesFileExists()) {
        //Création des colonnes si le fichier n'existe pas
        writeObject.writeColumnsName();
        cout << "oskour" << endl;
    }
    //Ecriture d'une nouvelle ligne
    writeObject.writeLine();
}
