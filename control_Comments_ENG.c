#include "control.h"	
#include "filter.h"	

u8 Flag_Target;// Target flag     
u8 temp1;      // Temporary variable                                         
float Voltage_Count,Voltage_All;  // Voltage variables
// Interrupt handler for EXTI lines 15 to 10
int EXTI15_10_IRQHandler(void) 
{    
	 if(INT==0)		
	{     
				EXTI->PR=1<<12;   // Clear interrupt pending flag                                                   		
			 Flag_Target=!Flag_Target;    // Toggle target flag                                           
		   if(delay_flag==1)
			 {
				 if(++delay_50==10)	 delay_50=0,delay_flag=0;                     
			 }
		  if(Flag_Target==1)                                               
			{ 
					Read_DMP();                                                 		
				  Voltage_All+=Get_battery_volt();  // Accumulate battery voltage                           
			    if(++Voltage_Count==100) Voltage=Voltage_All/100,Voltage_All=0,Voltage_Count=0;
			  	Key();//Scan key changes	
			return 0;	                                               
			}
		  Encoder_Balance=Read_Encoder(3);	// Read balance encoder value		
		  Encoder_Walk=Read_Encoder(2);     // Read walk encoder value
	  	Read_DMP();    // Read DMP data                                   
  		Led_Flash(100);  // Flash LED                                 
      Adjust=Get_Adc(Adjust_Ch);	// Read ADC value
      Middle=(Adjust-POTENTIOMETER)/392+3-Encoder_Balance/35;	// Calculate middle value	
		 	Get_RC(Flag_Direction);	// Get RC values

		  Motor_Walk=Walk_Control(Encoder_Walk,Target_Walk), // Calculate walk motor control value          
			Motor_Balance=Balance_Control(Pitch,gyro[1])+Position_Control(Encoder_Balance); // Calculate balance motor control value
  

		  Xianfu_Pwm(7000,110);    // Limit PWM values                             
			if(Turn_Off(Pitch,Voltage)==0)     
		  Set_Pwm(-Motor_Balance,Motor_Walk,Motor_Turn);   // Set PWM values                		 
			
 }
	 return 0;	 
} 
/**
 * Balance Control function
 * Calculates the balance control value based on the angle and gyro values.
 * 
 * @param Angle The current angle value.
 * @param Gyro The current gyro value.
 * @return The calculated balance control value.
 */
float Balance_Control(float Angle,float Gyro)
{  
   float Bias;
	 int balance;
	 Bias=Angle-Middle;  // Calculate the bias as the difference between the angle and the middle value
	 balance=Balance_KP*Bias+Gyro*Balance_KD;    // Calculate the balance control value using the bias and gyro values
	 return balance;
}

/**
 * Position Control function
 * Calculates the position control value based on the encoder value.
 * 
 * @param encoder The current encoder value.
 * @return The calculated position control value (velocity).
 */
 float Position_Control(int encoder)
{  
    static float Velocity,Encoder_Least,Movement,Encoder;
	  static float Encoder_Integral;  
    
		Encoder_Least=encoder; // Update the least significant encoder value     
		Encoder *= 0.7;		// Apply a low-pass filter to the encoder value                                                   
		Encoder += Encoder_Least*0.3;	                                      
 		Encoder_Integral +=Encoder;      // Update the encoder integral value                                 
		Encoder_Integral +=Movement;                                      
		if(Encoder_Integral>1500)  	Encoder_Integral=1500; // Limit the encoder integral value within a range              
		if(Encoder_Integral<-1500)	Encoder_Integral=-1500;              	
	  if(Flag_Stop)   Encoder_Integral=0; // Reset the encoder integral if the stop flag is set
    if (Flag_Stop)
		Velocity=Encoder*Position_KP+Encoder_Integral*Position_KI;    // Calculate the velocity using the encoder and integral values and the proportional and integral constants    	
	  if(Flag_Stop)   Velocity=0;    // Set the velocity to zero if the stop flag is set  
	  return Velocity;
}
// Walk control function
int Walk_Control (int Encoder,int Target)
{ 	
	 static int Bias,Pwm,Last_bias;// Calculate the bias as the difference between the encoder and target values
    Bias = Encoder - Target;// Update the PWM value using a proportional control with hysteresis
	 Bias=Encoder-Target;                
	 Pwm+=15*(Bias-Last_bias)+15*Bias;   
	 if(Pwm>7200)Pwm=7200;// Limit the PWM value within a specific range
    if (Pwm > 7200)
	 if(Pwm<-7200)Pwm=-7200;
	 Last_bias=Bias;	              // Update the last bias value for the next iteration      
	 return Pwm;                      // Return the calculated walk control value (PWM)
}   
}

// Set PWM values function
void Set_Pwm(int motor_a,int motor_b,int servo)
{  
	     float Turn;
	    
    	if(motor_a>0)			AIN1=7200,AIN2=7200-motor_a;  // Set PWM values for motor A
			else 	            AIN2=7200,AIN1=7200+motor_a;
		  
		  if(motor_b>0)			BIN1=1,BIN2=0;// Set PWM values for motor A
			else 	            BIN2=1,BIN1=0;
	    PWM=myabs(motor_b);// Set PWM value for motor B using absolute value of motor_b
	    
	    Turn=gyro[2]/50;// Calculate the turn value based on the gyro reading
	    SERVO=SERVO_MID+servo-Turn; // Set the servo PWM value with adjustments
}
// Limit PWM values function
void Xianfu_Pwm(int amplitude_motor,int amplitude_turn)
{	
    if(Motor_Balance<-amplitude_motor) Motor_Balance=-amplitude_motor;	// Limit motor balance PWM values
		if(Motor_Balance>amplitude_motor)  Motor_Balance=amplitude_motor;	
	  if(Motor_Walk<-amplitude_motor) Motor_Walk=-amplitude_motor;	// Limit motor walk PWM values
		if(Motor_Walk>amplitude_motor)  Motor_Walk=amplitude_motor;		 
	  if(Motor_Turn<-amplitude_turn)  Motor_Turn=-amplitude_turn;	// Limit motor turn PWM value
		if(Motor_Turn>amplitude_turn)  Motor_Turn=amplitude_turn;		

}
// Key function
void Key(void)
{	
	u8 tmp;
	tmp=click_N_Double(50); 
	if(tmp==1)Flag_Stop=!Flag_Stop;// If single click event Toggle the Flag_Stop value
	if(tmp==2)Flag_Show=!Flag_Show;// If double click event Toggle the Flag_Show value                
}

// Turn off function
u8 Turn_Off(float angle, int voltage)
{
	    u8 temp;
	    static u32 count;
      if(voltage<1100)count++;else 	count=0;	// Increment count if voltage is below 1100, otherwise reset count to 0
      if(Flag_Stop==1||Pitch<(Middle-30)||count>300||Pitch>(Middle+30)||Roll>15||Roll<-15)//// Check conditions for turning off control
			{	 
				Flag_Stop=1;				
				temp=1;          // Set the temporary flag to indicate control should be turned off                                  
				STBY=0;          // Disable control components
				PWM=0;
				BIN1=0;
				BIN2=0;
		    SERVO=SERVO_MID;
      }
			else
			{
				STBY=1; 
				temp=0;		    // Set the temporary flag to indicate control should not be turned off
	}	
			}
      return temp;			
}

// Absolute value function
u16 myabs(long int a)
{ 		   
	  long int temp;// Check if the input is negative
		if(a<0)  temp=-a;  // Compute the absolute value by negating the input
	  else temp=a;// Input is already positive, so the absolute value is the same as the input
	  return temp;// Return the absolute value as an unsigned 16-bit integer
}
// Get RC values function
void Get_RC(u8 mode)
{
		float step_servo=0.8;  // Step size for servo control
		float step_motor=0.2;  // Step size for motor control
	  float Max_Velocity;// Maximum velocity
		switch(mode)   //·½Ïò¿ØÖÆ
		{
		case 1:  Target_Walk-=step_motor;       Motor_Turn=Motor_Turn/1.01;   break;
		case 2:  Target_Walk-=step_motor;       Motor_Turn-=step_servo;   break;
		case 3:  Target_Walk=Target_Walk/1.03;  Motor_Turn-=step_servo;      break;
		case 4:  Target_Walk+=step_motor;       Motor_Turn-=step_servo;              break;
		case 5:  Target_Walk+=step_motor;       Motor_Turn=Motor_Turn/1.01;              break;
		case 6:  Target_Walk+=step_motor;       Motor_Turn+=step_servo;              break;
		case 7:  Target_Walk=Target_Walk/1.03;  Motor_Turn+=step_servo;                  break;
		case 8:  Target_Walk-=step_motor;       Motor_Turn+=step_servo;              break; 
		default: Target_Walk=Target_Walk/1.02;	Motor_Turn=Motor_Turn/1.01;	  break;	 
		}
		Max_Velocity=Velocity-myabs(Motor_Turn);
		if(Max_Velocity<14)Max_Velocity=14;
		if(Target_Walk<-Max_Velocity) Target_Walk=-Max_Velocity;	   
		if(Target_Walk>Max_Velocity/2)  Target_Walk=Max_Velocity/2;	     
		if(Motor_Turn<-110) Motor_Turn=-110;	   
		if(Motor_Turn>110)  Motor_Turn=110;	 	
}


