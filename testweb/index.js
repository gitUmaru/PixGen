
firebase.auth().onAuthStateChanged(function(user) {
  if (user) {
    
    document.getElementById("divison_user").style.display = "block";
    document.getElementById("loginarea").style.display = "none";

    var user = firebase.auth().currentUser;

    if(user != null){

      var email_id = user.email;
      document.getElementById("welcome_user_print").innerHTML = "Welcome User : " + email_id;

    }

  } else {
  

    document.getElementById("divison_user").style.display = "none";
    document.getElementById("loginarea").style.display = "block";

  }
});

function login(){

  var userEmail = document.getElementById("loginwithemail").value;
  var userPass = document.getElementById("loginwithpass").value;

  firebase.auth().signInWithEmailAndPassword(userEmail, userPass).catch(function(error) {
    
    var errorMessage = error.message;

    window.alert("Error : " + errorMessage);

 
  });

}


function register(){

  var email = document.getElementById("loginwithemail").value;
  var password = document.getElementById("loginwithpass").value;

  firebase.auth().createUserWithEmailAndPassword(email, password).catch(function(error) {
    // Handle Errors here.
    var errorCode = error.code;
    var errorMessage = error.message;
    window.alert("Error : " + errorMessage);
  });

  window.alert("Registered successfully!!");
 }

function logout(){
  firebase.auth().signOut();
}

function uploadImage(){
  const ref= firebase.storage().ref()

  const file= document.querySelector("#image").files[0]

  const name= new Date()+ '-'+file.name
  const metadata={
    contentType: file.type
  }
  const task= ref.child(name).put(file,metadata)
  task.then(snapshot => snapshot.ref.getDownloadURL())
  .then(url=> {
    console.log(url)
    alert("Image is uploaded successfully!")
    const image= document.querySelector('#imageupload')
    image.src= url
  })
}