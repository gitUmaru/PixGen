

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

function logout(){
  firebase.auth().signOut();
}
