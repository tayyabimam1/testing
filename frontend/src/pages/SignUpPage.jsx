import React, {useState} from 'react'
import {useNavigate} from 'react-router-dom'
import {firebaseAuth} from '../firebase/firebase'
import {createUserWithEmailAndPassword, sendEmailVerification} from 'firebase/auth'
import Button from 'react-bootstrap/Button'
import Toast from 'react-bootstrap/Toast'
import ToastContainer from 'react-bootstrap/ToastContainer'

const SignUpPage = () => {
  const [name, setName] = useState('')
  const [age, setAge] = useState()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')

  const [showToast, setShowToast] = useState(false)
  const [toastMessage, setToastMessage] = useState('')
  const [toastType, setToastType] = useState('success')

  const navigate = useNavigate();
  
  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      const userCredit = await createUserWithEmailAndPassword(firebaseAuth, email, password)
      const user = userCredit.user

      await sendEmailVerification(user)
      setToastMessage('User has successfully signed up! Please check your email for verification.')
      setToastType('success')
      setShowToast(true)
      setTimeout(() => navigate('/login'), 4000)
    }
    catch (error) {
      setToastMessage(error.message)
      setToastType('danger')
      setShowToast(true)
    }
  };

  return (
    <>
      <ToastContainer 
        position="top-end" 
        style={{ 
          position: 'fixed',
          top: 20,
          right: 20,
          zIndex: 9999 
        }}
      >
        <Toast 
          show={showToast} 
          onClose={() => setShowToast(false)} 
          bg={toastType}
          delay={4000} 
          autohide
        >
          <Toast.Header>
            <strong className="me-auto">{toastType === 'success' ? 'Success!' : 'Error'}</strong>
          </Toast.Header>
          <Toast.Body className={toastType === 'success' ? 'text-white' : ''}>
            {toastMessage}
          </Toast.Body>
        </Toast>
      </ToastContainer>

      <div className='custom-bg sign-up' style={{padding: '100px 0'}}>
        <div className="container text-center py-4 w-50" style={{backgroundColor: 'rgba(255, 255, 255, 0.05)'}}>
          <h1 className='text-color mb-4'>Sign Up</h1>
        <form className='d-flex flex-column gap-2' onSubmit={handleSubmit}>
          <label className='text-white mt-3'>Enter Your Name</label>
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            style={{backgroundColor: '#94A3B8'}} 
            className='w-50 mx-auto p-2' 
            type="text" 
            placeholder="Name"
            required 
          />

          <label className='text-white mt-3'>Enter Your Age</label>
          <input
            value={age}
            onChange={(e) => setAge(e.target.value)}
            style={{backgroundColor: '#94A3B8'}} 
            className='w-50 mx-auto p-2' 
            type="number" 
            min="13"
            max="100"
            placeholder="Age"
            required 
          />

          <label className='text-white mt-3'>Select Your Gender</label>
          <select 
            style={{backgroundColor: '#94A3B8'}} 
            className='w-50 mx-auto p-2'
            required
          >
            <option value="">Choose gender</option>
            <option value="male">Male</option>
            <option value="female">Female</option>
            <option value="other">Other</option>
          </select>
          
          <label className='text-white mt-3'>Enter Your Email</label>
          <input 
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            style={{backgroundColor: '#94A3B8'}} 
            className='w-50 mx-auto p-2' 
            type="email" 
            placeholder="example@email.com"
            required 
          />
          
          <label className='text-white mt-3'>Enter Your Password</label>
          <input 
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            style={{backgroundColor: '#94A3B8'}} 
            className='w-50 mx-auto p-2' 
            type="password" 
            placeholder="••••••••"
            minLength="5"
            required 
          />

          <Button 
            variant="primary"
            type="submit" 
            className="w-50 mx-auto mt-4 mb-3"
          >
            Sign Up
          </Button>
        </form>
      </div>
    </div>
    </>
  )
}

export default SignUpPage